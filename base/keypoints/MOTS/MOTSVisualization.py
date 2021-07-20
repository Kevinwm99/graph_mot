import sys
from Visualize import Visualizer
from mots_common.io import load_sequences, load_seqmap, load_txt
import pycocotools.mask as rletools
import glob
import os
import cv2
import colorsys
import numpy as np
import matplotlib.pyplot as plt

def apply_mask(image, mask, color, alpha=0.5):
	"""
	 Apply the given mask to the image.
	"""
	for c in range(3):
		image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c],image[:, :, c])
	return image


class MOTSVisualizer(Visualizer):

	def load(self, FilePath):
		return load_txt(FilePath)



	def drawResults(self, im = None, t = 0):
		self.draw_boxes = True

		sift = cv2.SIFT_create()
		for obj in self.resFile[t]:

			color = self.colors[obj.track_id % len(self.colors)]

			color = tuple([int(c*255) for c in color])


			if obj.class_id == 1:
				category_name = "Car"
			elif obj.class_id == 2:
				category_name = "Ped"
			else:
				category_name = "Ignore"
				color = (0.7*255, 0.7*255, 0.7*255)

			binary_mask = rletools.decode(obj.mask)
			posx, posy = np.where(binary_mask == 1)
			if obj.class_id == 1 or obj.class_id == 2:  # Don't show boxes or ids for ignore regions
				x, y, w, h = rletools.toBbox(obj.mask)

				pt1=(int(x),int(y))
				pt2=(int(x+w),int(y+h))

				category_name += ":" + str(obj.track_id)
				# cv2.putText(im, category_name, (int(x + 0.5 * w), int( y + 0.5 * h)), cv2.FONT_HERSHEY_TRIPLEX,self.imScale,color,thickness =2)
				if self.draw_boxes:
					cv2.rectangle(im,pt1,pt2,color,2)
					x = int(x)
					y = int(y)
					w = int(w)
					h = int(h)
					img = im[y:y+h, x:x+w, ::-1]
					kp = sift.detect(img)
					k_in = []
					k_out =[]
					for k in kp:
						# print(k.pt)
						kpx, kpy = k.pt
						# print(binary_mask)
						# print(binary_mask.shape)
						# cont = cv2.findContours(binary_mask*255, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1][0]
						contours, hierarchy = cv2.findContours(binary_mask[y:y+h,x:x+w].copy(),
																  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
																  offset=(0, 0))
						# print(contours)
						img_contours = np.zeros(binary_mask[y:y+h,x:x+w].shape)
						# draw the contours on the empty image
						# cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
						cv2.drawContours(img_contours, contours, -1, (127, 127, 127), 2)
						cv2.imwrite('/home/kevinwm99/MOT/MOTChallengeEvalKit/vid/cont-{}.png'.format(obj.track_id), img_contours)
						# exit()
						# print(contours[0])
						result = cv2.pointPolygonTest(contours[0], (int(kpy), int(kpx)), False)
						# print(result)
						# print(result)
						# exit()
						if result >=0:
							# cv2.circle(img, (kpx, kpy), radius=0, color=(0, 0, 255), thickness=-1)
							# img = cv2.drawKeypoints(img, k, outImage=None,
							# 					# flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
							# 					color=(0,0,255))
							k_in.append(k)
						else:
							# cv2.circle(img, (kpx, kpy), radius=0, color=(0, 255, 0), thickness=-1)
							# img = cv2.drawKeypoints(img, k, outImage=None,
							# 						# flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
							# 						color=(0, 255, 0))
							k_out.append(k)
					img = cv2.drawKeypoints(img, k_in, outImage=None,
																	# flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
																	color=(0, 0, 255))
					img = cv2.drawKeypoints(img, k_out, outImage=None,
																	# flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
																	color=(255, 0, 0))
					cv2.imwrite('/home/kevinwm99/MOT/MOTChallengeEvalKit/vid/kpbb-{}.png'.format(obj.track_id), img[:,:,::-1])
					plt.imshow(im[y:y+h, x:x+w, ::-1])
					plt.savefig("/home/kevinwm99/MOT/MOTChallengeEvalKit/vid/bb-{}.png".format(str(obj.track_id)))



			im = apply_mask(im, binary_mask, color)
		print(binary_mask)

		print(posx)
		print(posy)
		print(obj.mask)
		print(x, y, w, h)
		plt.imshow(im[:, :, ::-1])
		plt.savefig('/home/kevinwm99/MOT/MOTChallengeEvalKit/vid/test.png')
		exit()
		return im

if __name__ == "__main__":
	visualizer = MOTSVisualizer(
	seqName = "MOTS20-09",
	FilePath ="/home/kevinwm99/MOT/GCN/base/data/MOTS/train/MOTS20-09/gt/gt.txt",
	image_dir = "/home/kevinwm99/MOT/GCN/base/data/MOTS/train/MOTS20-09/img1",
	mode = "gt",
	output_dir  = "/home/kevinwm99/MOT/MOTChallengeEvalKit/vid")

	visualizer.generateVideo(
	        displayTime = True,
	        displayName = "seg",
	        showOccluder = True,
	        fps = 30 )
