#!/usr/bin/env python3

import math
import rospy

from tf.transformations import quaternion_from_euler
from tf2_ros import TransformBroadcaster
from pyproj import CRS, Transformer, Proj

from novatel_oem7_msgs.msg import INSPVA
from geometry_msgs.msg import PoseStamped, TwistStamped, Quaternion, TransformStamped

class Localizer:
    def __init__(self):

        # Parameters
        self.undulation = rospy.get_param('undulation')
        utm_origin_lat = rospy.get_param('utm_origin_lat')
        utm_origin_lon = rospy.get_param('utm_origin_lon')

        # Internal variables
        self.crs_wgs84 = CRS.from_epsg(4326)
        self.crs_utm = CRS.from_epsg(25835)
        self.utm_projection = Proj(self.crs_utm)

        # TODO 2: Create a coordinate transformer using self.crs_wgs84 and self.crs_utm.
        #         Use Transformer.from_crs(). Then transform the origin point (utm_origin_lat,
        #         utm_origin_lon) and store results as self.origin_x and self.origin_y.
        
        self.transformer = Transformer.from_crs(self.crs_wgs84, self.crs_utm)
        self.origin_x, self.origin_y = self.transformer.transform(utm_origin_lat, utm_origin_lon)

        # Subscribers
        rospy.Subscriber('/novatel/oem7/inspva', INSPVA, self.transform_coordinates)

        # Publishers
        self.current_pose_pub = rospy.Publisher('current_pose', PoseStamped, queue_size=10)
        self.current_velocity_pub = rospy.Publisher('current_velocity', TwistStamped, queue_size=10)
        self.br = TransformBroadcaster()

    def transform_coordinates(self, msg):
        # TODO 1: Print latitude and longitude from msg to verify data is received.
        pose_x, pose_y = self.transformer.transform(msg.latitude, msg.longitude)
        pose_x -= self.origin_x
        pose_y -= self.origin_y
        print(pose_x, pose_y)

        # TODO 3: Calculate orientation as a quaternion.
        #         - Get azimuth correction: self.utm_projection.get_factors(msg.longitude, msg.latitude).meridian_convergence
        #         - Subtract correction from msg.azimuth, convert to radians
        #         - Use convert_azimuth_to_yaw() to get yaw angle
        #         - Use quaternion_from_euler(0, 0, yaw) to get quaternion, create Quaternion object

        azimuth_correction = self.utm_projection.get_factors(msg.longitude, msg.latitude).meridian_convergence
        yaw = self.convert_azimuth_to_yaw(msg.azimuth - azimuth_correction)
        qx, qy, qz, qw = quaternion_from_euler(0, 0, yaw)
        orientation = Quaternion(qx, qy, qz, qw)

        # TODO 4: Create and publish a PoseStamped message on self.current_pose_pub:
        #         - header.stamp from msg.header.stamp, frame_id = "map"
        #         - position.x, position.y from transformed coordinates
        #         - position.z = msg.height - self.undulation
        #         - orientation from the quaternion

        current_pose_msg = PoseStamped()
        current_pose_msg.header.stamp = msg.header.stamp
        current_pose_msg.header.frame_id = "map"
        current_pose_msg.pose.position.x = pose_x
        current_pose_msg.pose.position.y = pose_y
        current_pose_msg.pose.position.z = msg.height - self.undulation
        current_pose_msg.pose.orientation = orientation
        self.current_pose_pub.publish(current_pose_msg)

        # TODO 5: Calculate velocity as norm of msg.north_velocity and msg.east_velocity.
        #         Create and publish a TwistStamped message on self.current_velocity_pub:
        #         - header.stamp from msg.header.stamp, frame_id = "base_link"
        #         - twist.linear.x = calculated velocity
        
        velocity = math.sqrt(msg.north_velocity**2 + msg.east_velocity**2)
        current_velocity_msg = TwistStamped()
        current_velocity_msg.header.stamp = msg.header.stamp
        current_velocity_msg.header.frame_id = "base_link"
        current_velocity_msg.twist.linear.x = velocity
        self.current_velocity_pub.publish(current_velocity_msg)

        # TODO 6: Create and publish a TransformStamped message using self.br.sendTransform():
        #         - header.stamp from msg.header.stamp, frame_id = "map"
        #         - child_frame_id = "base_link"
        #         - transform.translation from position (x, y, z)
        #         - transform.rotation from orientation quaternion

        transform_msg = TransformStamped()
        transform_msg.header.stamp = msg.header.stamp
        transform_msg.header.frame_id = "map"
        transform_msg.child_frame_id = "base_link"
        transform_msg.transform.translation.x = pose_x
        transform_msg.transform.translation.y = pose_y
        transform_msg.transform.translation.z = msg.height - self.undulation
        transform_msg.transform.rotation = orientation
        self.br.sendTransform(transform_msg)

    @staticmethod
    def convert_azimuth_to_yaw(azimuth):
        """
        Converts azimuth to yaw. Azimuth is CW angle from the north. Yaw is CCW angle from the East.
        :param azimuth: azimuth in radians
        :return: yaw in radians
        """
        yaw = -azimuth + math.pi / 2
        # Clamp within 0 to 2 pi
        if yaw > 2 * math.pi:
            yaw = yaw - 2 * math.pi
        elif yaw < 0:
            yaw += 2 * math.pi

        return yaw

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('localizer')
    node = Localizer()
    node.run()