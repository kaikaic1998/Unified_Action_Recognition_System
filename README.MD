(**Code with Papers: Track & Detection Search**):(https://paperswithcode.com/search?q_meta=&q_type=&q=tracking+detection)

(**Code with Papers: Multi-Object Tracking**):(https://paperswithcode.com/task/multi-object-tracking/latest)

•	这个方向需要
o	检测追踪骨骼识别行动识别
o	只训练最后一个动作识别，别的用现成的就可以
•	检测和追踪你可以用yolo，然后骨骼识别用什么模型，以及行动识别用什么模型都需要做调查
•	先调查确定用什么模型，然后再开始收集数据训练，都需要什么模型，现在最先进的模型有哪些等等
Person re-identification (ReID) – An embedding model
OpenCV tracking human from given bounding box and assign ID to each detected human
YOLOX - Apache-2.0 license

Object detection & tracking
•	Hybrid-SORT: Weak Cues Matter for Online Multi-Object Tracking – 2023/8/1
o	GitHub - GPL-3.0 license
o	Two steps
1.	Use YOLOX model as detector
2.	Use BoT-SORT for ReID
•	Yolov7 + StrongSORT with OSNet GitHub – 2022 - GPL-3.0 license
•	ByteTrack: Multi-Object Tracking by Associating Every Detection Box – 2021/10/13
o	GitHub – MIT license
o	Two steps
1.	Use YOLOX as detector
2.	Combining BYTE for ReID
•	Towards Real-Time Multi-Object Tracking – 2019/9/27
o	GitHub – MIT license
o	Joint Detection Embedding
	Simultaneously output the location and appearance embeddings of targets in a single forward pass, as a single unified model
	Use Darknet-53 as backbone network (similar to YOLO)
•	BoT-SORT: Robust Associations Multi-Pedestrian Tracking – 2022/6/29
o	GitHub – MIT license
o	It is only a tracker (ReID). Relying YOLOX for detect and provide bounding box
•	Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking – 2022/3/27
o	GitHub – MIT license


Pose Estimation (usually works in 2 stages)
1.	Person detection
2.	Keypoints localization
•	Based on which stage comes first, they can be categorized into the Top-down and Bottom-up approaches.
o	Top-down approach
	The person is detected first then the landmarks are localized for each person
	More the number of persons, the more the computational complexity. 
	These approaches are scale invariant. They perform well on popular benchmarks in terms of accuracy. However, due to the complexity of these models, achieving real-time inference is computationally expensive.
o	Bottom-up approach
	It finds identity-free landmarks (keypoints) of all the persons in an image at once, followed by grouping them into individual persons. 
	A probabilistic map called heatmap is used by these approaches to estimate the probability of every pixel containing a particular landmark (keypoint). 
	With the help of Non-Maximum Suppression, the best landmark is filtered. These are less complex compared to Top-down methods but at the cost of reduced accuracy
•	AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time – 2022/11/7
o	GitHub - For non-commercial research use by academic or non-profit organizations only
•	Human Pose as Compositional Tokens – 2023/3/21
o	GitHub – MIT license
•	RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose – 2023/7/3
o	GitHub - Apache-2.0 license 
•	ViTPose+: Vision Transformer Foundation Model for Generic Body Pose Estimation 2022/12/7
o	GitHub - Apache-2.0 license
•	YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss – 2022/4/14
o	GitHub - GPL-3.0 license
o	YOLOv5 based (The pose implementation branch in YOLOv7 repo is also based on this)
•	

Fall Detection using Pose Estimation
YOLOv7 Pose vs MediaPipe in Human Pose Estimation


