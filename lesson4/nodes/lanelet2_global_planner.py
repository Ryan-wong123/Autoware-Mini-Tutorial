#!/usr/bin/env python3

import numpy as np
import rospy
from threading import Lock

from geometry_msgs.msg import PoseStamped
from autoware_mini.msg import Path, Waypoint

import lanelet2
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findNearest


class GlobalPlanner:
    def __init__(self):

        # Parameters
        lanelet2_map_path = rospy.get_param("~lanelet2_map_path")
        self.speed_limit = float(rospy.get_param("~speed_limit"))

        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")

        self.output_frame = rospy.get_param("lanelet2_global_planner/output_frame")
        self.distance_to_goal_limit = rospy.get_param("lanelet2_global_planner/distance_to_goal_limit")

        # Load Lanelet2 map
        if coordinate_transformer == "utm":
            projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
        else:
            raise RuntimeError('Only "utm" is supported for lanelet2 map loading')
        self.lanelet2_map = load(lanelet2_map_path, projector)

        # Create traffic rules and routing graph.
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                              lanelet2.traffic_rules.Participants.VehicleTaxi)
        self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)

        # Internal variables
        self.lock = Lock()
        self.current_location = None
        self.goal_point = None

        # Publishers
        self.global_path_pub = rospy.Publisher('global_path', Path, latch=True, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

    def goal_callback(self, msg):
        with self.lock:
            self.goal_point = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)

        if self.current_location is None:
            return

        # Log the received goal position coordinates.
        rospy.loginfo("%s - goal position (%f, %f, %f) in %s frame", rospy.get_name(),
                    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                    msg.header.frame_id)
                    
        # Find the route from current location to goal.
        start_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.current_location, 1)[0][1]
        goal_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.goal_point, 1)[0][1]

        route = self.graph.getRoute(start_lanelet, goal_lanelet, 0, False)
        if route is None:
            rospy.logwarn("%s - No route found to goal position", rospy.get_name())
            return

        path = route.shortestPath()
        if path is None:
            rospy.logwarn("%s - No shortest path found to goal position", rospy.get_name())
            return

        path_no_lane_change = path.getRemainingLane(start_lanelet)

        waypoints = self.convert_laneletseq_to_waypoints_list(path_no_lane_change)
        self.publish_lane_from_waypoints_list(waypoints)

    def current_pose_callback(self, msg):
        with self.lock:
            self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)

        if self.goal_point is None:
            return

        distance_to_goal = np.linalg.norm(
            np.array([self.current_location.x - self.goal_point.x,
                      self.current_location.y - self.goal_point.y])
        )

        if distance_to_goal <= self.distance_to_goal_limit:
            rospy.loginfo("%s - goal position reached", rospy.get_name())
            self.publish_lane_from_waypoints_list([])
            with self.lock:
                self.goal_point = None
            return

    def convert_laneletseq_to_waypoints_list(self, laneletseq):
        waypoints = []
        lanelet_start_indices = []

        for j, lanelet in enumerate(laneletseq):
            try:
                speed_ref_kmh = lanelet.attributes["speed_ref"]
            except KeyError:
                speed_ref_kmh = None

            if speed_ref_kmh is not None:
                speed = min(float(speed_ref_kmh), self.speed_limit) / 3.6
            else:
                speed = self.speed_limit / 3.6

            lanelet_start_indices.append(len(waypoints))

            for i, point in enumerate(lanelet.centerline):
                if i == 0 and j != 0:
                    continue

                waypoint = Waypoint()
                waypoint.position.x = point.x
                waypoint.position.y = point.y
                waypoint.position.z = point.z
                waypoint.speed = speed
                waypoints.append(waypoint)

        if self.goal_point is not None and len(waypoints) >= 2:
            goal_xy = np.array([self.goal_point.x, self.goal_point.y], dtype=float)
            best_segment_index = None
            best_t = None
            best_point = None
            best_distance = None

            for idx in range(len(waypoints) - 1):
                p1 = np.array([waypoints[idx].position.x, waypoints[idx].position.y], dtype=float)
                p2 = np.array([waypoints[idx + 1].position.x, waypoints[idx + 1].position.y], dtype=float)
                segment_vec = p2 - p1
                segment_length_sq = np.dot(segment_vec, segment_vec)

                if segment_length_sq < 1e-12:
                    continue

                t = np.dot(goal_xy - p1, segment_vec) / segment_length_sq
                t = min(max(t, 0.0), 1.0)
                projected_point = p1 + t * segment_vec
                distance = np.linalg.norm(projected_point - goal_xy)

                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_segment_index = idx
                    best_t = t
                    best_point = projected_point

            if best_segment_index is not None and best_point is not None:
                projected_waypoint = Waypoint()
                projected_waypoint.position.x = float(best_point[0])
                projected_waypoint.position.y = float(best_point[1])
                projected_waypoint.position.z = float(
                    waypoints[best_segment_index].position.z
                    + best_t * (
                        waypoints[best_segment_index + 1].position.z - waypoints[best_segment_index].position.z
                    )
                )
                projected_waypoint.speed = 0.0

                goal_waypoint = Waypoint()
                goal_waypoint.position.x = float(self.goal_point.x)
                goal_waypoint.position.y = float(self.goal_point.y)
                goal_waypoint.position.z = float(projected_waypoint.position.z)
                goal_waypoint.speed = 0.0

                truncated_waypoints = []
                for idx, waypoint in enumerate(waypoints):
                    if idx <= best_segment_index:
                        truncated_waypoints.append(waypoint)
                    else:
                        break

                truncated_waypoints.append(projected_waypoint)
                truncated_waypoints.append(goal_waypoint)
                with self.lock:
                    self.goal_point = BasicPoint2d(goal_waypoint.position.x, goal_waypoint.position.y)
                return truncated_waypoints

        return waypoints

    def publish_lane_from_waypoints_list(self, waypoints):
        lane = Path()
        lane.header.frame_id = self.output_frame
        lane.header.stamp = rospy.Time.now()
        lane.waypoints = waypoints
        self.global_path_pub.publish(lane)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('global_planner')
    node = GlobalPlanner()
    node.run()
