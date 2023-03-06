import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image
import cv2

from functools import reduce
import plotly.express as px

import itertools

st.header("Algo Sculpor")

st.write("This framework is used to turned at least two 2D images into 3D images and compute it volume")

st.subheader("Loading images")

n_img = st.number_input("how many images do you have?", min_value = 2, max_value = 4, key = "n_img")

img_cols = st.columns(n_img)
img_type_cols = st.columns(n_img)
img_type_options = ["top", "front", "left side", "right side"]

imgs = []
img_types = []

for i in range(n_img):
	imgs.append(img_cols[i].file_uploader("Image "+str(i)))


def filter_selectbox(lst1, lst2):
	return list(set(lst2)-set(lst1))
	
for i in range(n_img):
	img_types.append(img_type_cols[i].selectbox("Type, Image "+str(i),
	                                             filter_selectbox(img_types, img_type_options)))

####
def save_image(img, name):
	filepath = 'app_lab/' + name + '.jpg'
	Image.open(img).save(filepath)
	return filepath
	
def get_image(img, imgname):
	return cv2.imread(save_image(img, imgname), cv2.IMREAD_GRAYSCALE)


def plot_imgs(imgs, resize_fig = (10,10)):
	ncols = len(imgs)
	plt.figure(figsize = resize_fig)
	fig, axs = plt.subplots(1, ncols)
	
	i = 0
	for k,v in imgs.items():
		axs[i].imshow(v, cmap = 'gray')
		axs[i].set_title(k)
		i = i + 1
	
	return fig

def extract_xy_image(img):
	xy_img = {'x' : [], 'y' : [], 'min' : list(img.shape), 'max' : [0, 0]}
	
	for i in range(img.shape[0]):
		for k in range(img.shape[1]):
			if (img[i,k] == 0):
				xy_img['y'].append(i)
				xy_img['x'].append(k)
				
				if xy_img['min'][0] > i :
					xy_img['min'][0] = i
					
				if xy_img['min'][1] > k :
					xy_img['min'][1] = k
					
				if xy_img['max'][0] < i :
					xy_img['max'][0] = i
					
				if xy_img['max'][1] < k :
					xy_img['max'][1] = k
					
	return xy_img
	
	
def extruder(xy_img, minx, maxy, fix = 'x'):
	moves = ['x', 'y', 'z']
	moves.remove(fix)
	
	xyz_img = {'x' : [], 'y' : [], 'z' : []}
	
	for i in range(len(xy_img['x'])):
		# with 2D_xy if 3D_y is fixed then 3D_x = 2D_x and 3D_z = 2D_y 
		vals = [xy_img['x'][i], xy_img['y'][i]] #TODO: check if it's right doing this
		for k in range(minx, maxy):
			xyz_img[fix].append(k)
			for j, e in enumerate(moves):
				xyz_img[e].append(vals[j])
		
	return xyz_img
	
def intersect_df(df1, df2):
	return pd.merge(df1, df2, how = 'inner', on = ['x', 'y', 'z'])
	
def intersect_3D_imgs(xyz_imgs):
	
	dfs = []
	for e in xyz_imgs.values() :
		dfs.append(pd.DataFrame(e))
		
	return reduce(lambda df_left, df_right : pd.merge(df_left, df_right, how = 'inner', on = ['x', 'y', 'z']), dfs)
		
def plot_xy_imgs(xy_imgs, resize_fig = (10, 10)):
    
	ncols = len(xy_imgs)
	
	plt.figure(figsize = resize_fig)
	fig, axs = plt.subplots(1, ncols)
    
	i = 0
	for e in xy_imgs:
		axs[i].scatter(e['x'], e['y'])
		i += 1
        
	return fig
    
def run(imgs):
	n_px = imgs['top'].shape[0]
	
	xyz_imgs = {}
	fixes = {"top" : 'z', "front" : 'y' , "left side" : '-x', "right side" : 'x'}
	
	for k, v in imgs.items():
		xy_img = extract_xy_image(v)
		
		xyz_img = extruder(xy_img, 0, n_px, fix = fixes[k])
		xyz_imgs[k] = xyz_img
				
	return intersect_3D_imgs(xyz_imgs)
		

def project_3D_2_2D(df, axis_limit = 400, resize_fig = (10, 10)) :
	cols = df.columns
	col_pairs = list(itertools.combinations(cols, 2))
	ncols = len(col_pairs)
    
	plt.figure(figsize = resize_fig)
	fig, axs = plt.subplots(1, ncols)
    
	i = 0
	for e in col_pairs:
		new_df = df.loc[:,list(e)].drop_duplicates()
		axs[i].scatter(new_df.iloc[:,0], new_df.iloc[:,1])
		axs[i].set_title('.'.join(list(e)))
		axs[i].set_xlim(0, axis_limit)
		axs[i].set_ylim(0, axis_limit)
		axs[i].set_aspect('equal', adjustable='box')
		i += 1
        
	return fig
    

def plot_xyzimg(df, axis_limit = 400):
	fig = plt.figure(figsize=(12, 12))
	ax = fig.add_subplot(projection='3d')
	ax.scatter(df.x, df.y, df.z)
	ax.set_xlim(0, axis_limit)
	ax.set_ylim(0, axis_limit)
	ax.set_aspect('equal', adjustable='box')
	return fig

def voxel2cm3(vx, dpi = 300):
	return 2.54 * 2.54 * 2.54 * vx / dpi / dpi / dpi

if len(imgs) >= 2:

	st.subheader("Show images")

	the_images = {}
	for i in range(n_img):
		if imgs[i] is not None:
			the_images[img_types[i]] = get_image(imgs[i], img_types[i]) 

	if len(the_images) >= 2:
		st.pyplot(plot_imgs(the_images))
	
	
		st.subheader("Compute the 3D Image")
		
		if st.button('Run') : 
			with st.spinner('Running') :
				img3D = run(the_images)
				st.pyplot(plot_xyzimg(img3D))
				#fig = px.scatter_3d(img3D, x = 'x', y = 'y', z = 'z')
				#st.plotly_chart(fig, use_container_width = True)
				st.pyplot(project_3D_2_2D(img3D))
				st.write("Volume = " + str(len(img3D)) + " Voxels = " + str(voxel2cm3(len(img3D))) + " cm3" )
	
	
	
