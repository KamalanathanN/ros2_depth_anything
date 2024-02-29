import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import torch
import cv2
import numpy as np
from torchvision.transforms import Compose

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

class DepthEstimator(Node):
    def __init__(self):
        super().__init__('depth_estimator')
        
        self.declare_parameter('encoder', 'vits')
        
        encoder = self.get_parameter('encoder').get_parameter_value().string_value
        self.depth_anything = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to('cuda' if torch.cuda.is_available() else 'cpu')
        self.depth_anything.eval()

        self.transform = Compose([
            Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
                   resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        self.br = CvBridge()
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        ) 
        self.subscriber = self.create_subscription(Image, '/zed_node/rgb/image_rect_color', self.image_callback, 10)
        #self.subscriber = self.create_subscription(Image, '/camera/id_0/image_color', self.image_callback, qos_profile)
        self.publisher = self.create_publisher(Image, 'depth_anything_image_topic', 10)

    def image_callback(self, msg):
        raw_image = self.br.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        h, w = image.shape[:2]

        image = self.transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        with torch.no_grad():
            depth = self.depth_anything(image)
        
        depth = torch.nn.functional.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
        depth = depth.cpu().numpy().astype(np.uint8)
        depth_image = self.br.cv2_to_imgmsg(depth, encoding="mono8")
        
        self.publisher.publish(depth_image)


def main(args=None):
    rclpy.init(args=args)
    depth_estimator = DepthEstimator()
    rclpy.spin(depth_estimator)
    depth_estimator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
