# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

from app.graphical_object import blueprint
from flask import render_template, redirect, url_for
from flask_login import login_required, current_user
from app import login_manager
from jinja2 import TemplateNotFound
from app import db
from sqlalchemy import desc, asc
from flask import request
import json
from sqlalchemy.sql import func
from werkzeug.utils import secure_filename
import os
from app.graphical_object.utils import convert_hash
from app.graphical_object.utils import hamming
from app.graphical_object.utils import dhash
from imutils import paths
from imutils import url_to_image
import pickle
import vptree
import cv2
import time
from zipfile import ZipFile
import io


basedir = os.path.abspath(os.path.dirname(__file__))
result = {'query':'web/E2.png'}
detect = []
@blueprint.route('/graphical_object/indexing')
@login_required
def indexing():
    # check exist user
    if not current_user.is_authenticated:
        return redirect(url_for('base_blueprint.login'))
    return render_template('indexing.html')

@blueprint.route('/graphical_object/index_images', methods=['POST'])
@login_required
def index_images():
    try:
        basedir = os.path.abspath(os.path.dirname(__file__))
        # load the input query image
        file =  request.files.to_dict()
        # Store Pdf with convert_from_path function
        filename = secure_filename(file['file'].filename)
        upload_path = os.path.join(basedir, 'static', 'images')
        # specifying the zip file name
        file_path = os.path.join(upload_path, filename)
        #file['file'].save(file_path)
        z = ZipFile(file['file'].stream._file)

        print('Extracting all the files now...')
        z.extractall(upload_path)
        print('Done!')

        # # delete the zip file
        # os.remove(file_path)

        # grab the paths to the input images and initialize the dictionary
        # of hashes

        imagePaths = list(paths.list_images(os.path.join(basedir, 'static','images')))
        hashes = {}
        print(imagePaths)
        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
	        # load the input image
	        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
	        image = cv2.imread(imagePath)
	        # compute the hash for the image and convert it
	        h = dhash(image)
	        h = convert_hash(h)
	        # update the hashes dictionary
	        l = hashes.get(h, [])
	        l.append(imagePath)
	        hashes[h] = l

        # build the VP-Tree
        print("[INFO] building VP-Tree...")
        points = list(hashes.keys())
        tree = vptree.VPTree(points, hamming)

        # serialize the VP-Tree to disk
        print("[INFO] serializing VP-Tree...")
        tree_path = os.path.join(basedir, 'static','indexing','vptree.pickle')
        f = open(tree_path, "wb")
        f.write(pickle.dumps(tree))
        f.close()
        # serialize the hashes to dictionary
        print("[INFO] serializing hashes...")
        hash_path = os.path.join(basedir, 'static','indexing','hashes.pickle')
        f = open(hash_path, "wb")
        f.write(pickle.dumps(hashes))
        f.close()

        return json.dumps({'status': 'OK','message':'The Result of the indexing is saved!{}'.format(file_path)})
    except:
        return json.dumps({'status': 'OK','message':'Error in the processing of the indexing. Please try again!'})

@blueprint.route('/graphical_object/searching')
@login_required
def searching():
    # check exist user
    if not current_user.is_authenticated:
        return redirect(url_for('base_blueprint.login'))
    return render_template('searching.html', result = result)

@blueprint.route('/graphical_object/search', methods=['POST'])
@login_required
def search():
    # load the VP-Tree and hashes dictionary
    print("[INFO] loading VP-Tree and hashes...")
    tree_path = os.path.join(basedir, 'static','indexing','vptree.pickle')
    tree = pickle.loads(open(tree_path, "rb").read())

    hash_path = os.path.join(basedir, 'static','indexing','hashes.pickle')
    hashes = pickle.loads(open(hash_path, "rb").read())

    # load the input query image
    form = request.form.to_dict()
    file = request.files.to_dict()

    if form['link'] == '':
        filename = secure_filename(file['file'].filename)
        image_query = os.path.join(basedir, 'static', 'queries', filename)
        file['file'].save(image_query)
        query = '/queries/{}'.format(filename)
        image = cv2.imread(image_query)
    else:
        image_query = form['link']
        image = url_to_image(image_query)

    # compute the hash for the query image, then convert it
    queryHash = dhash(image)
    queryHash = convert_hash(queryHash)

    # perform the search
    print("[INFO] performing search...")
    start = time.time()
    results = tree.get_all_in_range(queryHash, 20)
    results = sorted(results)
    end = time.time()
    print("[INFO] search took {} seconds".format(end - start))
    response = []
    # loop over the results
    for (d, h) in results:
	    # grab all image paths in our dataset with the same hash
        paths = ['/' + '/'.join(path.split('\\')[5:]) for path in hashes.get(h, [])]

        r = {'score' : (20-d)*5 , 'hash': h, 'paths': paths }
	    # print("[INFO] {} total image(s) with d: {}, h: {}".format(
		#     len(resultPaths), d, h))
	    # print(resultPaths)
        response.append(r)

    #res = json.dumps({'status': 'OK','message':'The Result of the search is displayed!', 'response' : response}, ensure_ascii=False)

    global result
    result = {'response': response, 'time': round(end - start, 5), 'query': query}
    return json.dumps({'status': 'OK','message':'The Result of the search is displayed!'})

@blueprint.route('/graphical_object/detect', methods=['POST'])
@login_required
def detect():
    # check exist user
    if not current_user.is_authenticated:
        return redirect(url_for('base_blueprint.login'))

    return json.dumps({'status': 'OK','message':'The Result of the detection is displayed!'})

@blueprint.route('/graphical_object/detection')
@login_required
def detection():
    # check exist user
    if not current_user.is_authenticated:
        return redirect(url_for('base_blueprint.login'))

    return render_template('detection.html', detect = detect)

@blueprint.route('graphical_object/<template>')
def route_template(template):

    if not current_user.is_authenticated:
        return redirect(url_for('base_blueprint.login'))

    try:

        return render_template(template + '.html')

    except TemplateNotFound:
        return render_template('page-404.html'), 404

    except:
        return render_template('page-500.html'), 500
@blueprint.route('graphical_object/<static>')
def route_static(static):

    if not current_user.is_authenticated:
        return redirect(url_for('base_blueprint.login'))

    try:

        return render_template(static + '.html')

    except TemplateNotFound:
        return render_template('page-404.html'), 404

    except:
        return render_template('page-500.html'), 500