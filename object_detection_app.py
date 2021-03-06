"""
An experimentation script to detect objects in real time using the webcam.
An excersise in the usage of tensor flow, and more advanced python libraries.
The current distro of the code works on Windows 10 and Anacaonda 3.5
Based on source code provided by Dat Tran

2018/08/10
author: @Henri De Boever

"Speed/accuracy trade-offs for modern convolutional object detectors."
Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z,
Song Y, Guadarrama S, Murphy K, CVPR 2017
"""

import os, cv2, time, argparse, multiprocessing, sys
import numpy as np, tensorflow as tf
from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class ObjectDetector():

	def __init__(self, MODEL):

		# Initialize Current Path
		self.CWD_PATH = os.getcwd()

		# Path to frozen detection graph. This is the actual model that is used for the object detection.
		self.MODEL_DIR_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
		self.MODEL =  MODEL
		self.PATH_TO_CKPT = os.path.join(self.CWD_PATH, 'object_detection', self.MODEL_DIR_NAME, self.MODEL)

		# List of the strings that is used to add the correct label for each box.
		self.PATH_TO_LABELS = os.path.join(self.CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
		self.NUM_CLASSES = 90

		# Loading label map
		self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
		self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes = self.NUM_CLASSES, use_display_name = True)
		self.category_index = label_map_util.create_category_index(self.categories)

	def detect_objects(self, image_np, sess, detection_graph):

		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image_np, axis = 0)
		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

		# Each box represents a part of the image where a particular object was detected.
		boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

		# Each score represent the level of confidence for each of the objects.
		# Score is shown on the result image, together with the class label.
		scores = detection_graph.get_tensor_by_name('detection_scores:0')
		classes = detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = detection_graph.get_tensor_by_name('num_detections:0')

		# Actual detection
		(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict = {image_tensor: image_np_expanded})

		# Visualization of the results of a detection.
		vis_util.visualize_boxes_and_labels_on_image_array(
			image_np,
			np.squeeze(boxes),
			np.squeeze(classes).astype(np.int32),
			np.squeeze(scores),
			self.category_index,
			use_normalized_coordinates = True,
			line_thickness = 4)

		# ----- For printing to the console during execution frames -----
		# Zip the object index with it's associated percentage prediction if the prediction is greater than 0.5
		classes_with_predictions = list(zip(np.squeeze(classes).astype(np.int32).tolist(), [item for item in np.squeeze(scores).tolist() if item > 0.5]))
		print(classes_with_predictions)

		# For each item in the reduced list, find its name in the self.category_index dictionary, and output it to the console
		for object_tuple in classes_with_predictions:
			for key, item in self.category_index.items():
				if (key == object_tuple[0]):
					print(item['name'], object_tuple[1])
					# Break once it is found to stop needless searching
					break
		return image_np

	def worker(self, input_q, output_q):
		print("Initialized a worker!")
		# Load a (frozen) Tensorflow model into memory.
		detection_graph = tf.Graph()
		# print(dir(detection_graph))
		with detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

			sess = tf.Session(graph = detection_graph)

		fps = FPS().start()
		while True:
			fps.update()
			frame = input_q.get()
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			output_q.put(self.detect_objects(frame_rgb, sess, detection_graph))

		fps.stop()
		sess.close()

def main(argv):

	print("\n---------- Starting object detection ----------\n")

	# Instantiate an ObjectDetector class object
	# Takes the name of the model graph as an argument
	ObjectFinder = ObjectDetector('frozen_inference_graph.pb')

	# Initialize a parser object
	parser = argparse.ArgumentParser()
	parser.add_argument('-src', '--source', dest = 'video_source', type=int, default = 0, help = 'Device index of the camera.')
	parser.add_argument('-wd', '--width', dest = 'width', type = int, default = 1080, help = 'Width of the frames in the video stream.')
	parser.add_argument('-ht', '--height', dest = 'height', type = int, default = 720, help = 'Height of the frames in the video stream.')
	parser.add_argument('-num-w', '--num-workers', dest = 'num_workers', type = int, default = 4, help = 'Number of workers.')
	parser.add_argument('-q-size', '--queue-size', dest = 'queue_size', type = int, default = 25, help = 'Size of the queue.')
	args = parser.parse_args()

	# Initialize a logger object
	logger = multiprocessing.log_to_stderr()
	logger.setLevel(multiprocessing.SUBDEBUG)
	input_q = Queue(maxsize = args.queue_size)
	output_q = Queue(maxsize = args.queue_size)
	pool = Pool(args.num_workers, ObjectFinder.worker, (input_q, output_q))
	video_capture = WebcamVideoStream(src = args.video_source, width = args.width, height = args.height).start()

	# ------------------------------Control Loop ------------------------------
	fps = FPS().start()
	# fps._numFrames < 120
	frame_number = 0
	while True:
		frame_number += 1
		# Frame is a numpy nd array
		frame = video_capture.read()
		input_q.put(frame)
		t = time.time()
		output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
		cv2.imshow('Video', output_rgb)
		fps.update()
		print("[INFO] elapsed time: {0:.3f}\nFrame number: {1}-------------------------------".format((time.time() - t), frame_number))
		if (cv2.waitKey(1) & 0xFF == ord('q')):
			break
	fps.stop()
	print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
	print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
	pool.terminate()
	video_capture.stop()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
