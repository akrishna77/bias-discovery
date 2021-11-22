import os
import sys
import logging
import time
from flask import request, render_template, send_from_directory, session, url_for, redirect

from app import app
from pathlib import Path

from argparse import Namespace

from experiments.celeba.plot import plot as plot_celeba
from experiments.coco.plot import plot as plot_coco
from experiments.celeba.discover import convert_to_dict as convert_to_dict_celeba
from experiments.celeba.discover import celeba_nn, celeba_dist_nn
from experiments.coco.discover_multi import convert_to_dict as convert_to_dict_coco

APP_ROOT = Path(__file__).resolve().parent
PROJ_ROOT = Path(__file__).resolve().parent.parent.parent

l = logging.getLogger(__name__)
l.setLevel(logging.INFO)

c_handler = logging.StreamHandler(sys.stdout)
c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)

l.addHandler(c_handler)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_model")
def upload_model():
    return render_template("upload_model.html")


@app.route("/upload_json")
def upload_json():
    return render_template("upload_json.html")


@app.route("/get_details")
def get_details():
    return render_template("get_details.html")


@app.route("/plot_cluster", methods=["POST"])
def plot_cluster():
    json_target = os.path.join(APP_ROOT, "files")
    model_target = os.path.join(APP_ROOT, "models")

    m = request.form.get("model-name")
    model_destination = "/".join([model_target, m])

    j = request.form.get("json-name")
    json_destination = "/".join([json_target, j])

    c_id = request.form.get("cluster-id")[7:]

    output_file = f"image-{c_id}-{time.strftime('%H%M%S')}.png"

    if request.form.get("dataset-select") == "COCO":
        a = Namespace(
            model=model_destination,
            model_type="densenet",
            model_specific_type="densenet201",
            gradcam_layer="module_features_denseblock4_denselayer32",
            biased_category=request.form.get("gradcam-class"),
            tree_json=json_destination,
            cluster_id=c_id,
            output_path=os.path.join(APP_ROOT, "images", output_file),
        )

        plot_coco(a, l)

    elif request.form.get("dataset-select") == "CelebA":
        a = Namespace(
            dataroot=f"{PROJ_ROOT}/celeba/img_align_celeba/",
            csv_file=f"{PROJ_ROOT}/celeba/list_attr_celeba.txt",
            delimiter_type="space",
            model=model_destination,
            gradcam_layer="layer4",
            batch_size=32,
            model_type="resnet",
            tree_json=json_destination,
            cluster_id=c_id,
            output_path=os.path.join(APP_ROOT, "images", output_file),
        )

        plot_celeba(a, l)

    session["output_file"] = output_file
    return redirect(url_for("display_cluster"))


@app.route("/plot_nn", methods=["POST"])
def plot_nn():
    json_target = os.path.join(APP_ROOT, "files")
    model_target = os.path.join(APP_ROOT, "models")

    m = request.form.get("model-name")
    model_destination = "/".join([model_target, m])

    j = request.form.get("json-name")
    json_destination = "/".join([json_target, j])

    c_id = request.form.get("cluster-id")[7:]

    output_file = f"image-{c_id}-{time.strftime('%H%M%S')}.png"

    if request.form.get("dataset-select") == "CelebA":
        new_c_id = celeba_nn(json_destination, c_id)

        a = Namespace(
            dataroot=f"{PROJ_ROOT}/celeba/img_align_celeba/",
            csv_file=f"{PROJ_ROOT}/celeba/list_attr_celeba.txt",
            delimiter_type="space",
            model=model_destination,
            gradcam_layer="layer4",
            batch_size=32,
            model_type="resnet",
            tree_json=json_destination,
            cluster_id=new_c_id,
            output_path=os.path.join(APP_ROOT, "images", output_file),
        )

        plot_celeba(a, l)

    session["output_file"] = output_file
    return redirect(url_for("display_cluster"))


@app.route("/plot_dist_nn", methods=["POST"])
def plot_dist_nn():
    json_target = os.path.join(APP_ROOT, "files")
    model_target = os.path.join(APP_ROOT, "models")

    m = request.form.get("model-name")
    model_destination = "/".join([model_target, m])

    j = request.form.get("json-name")
    json_destination = "/".join([json_target, j])

    c_id = request.form.get("cluster-id")[7:]

    output_file = f"image-{c_id}-{time.strftime('%H%M%S')}.png"

    if request.form.get("dataset-select") == "CelebA":

        new_c_id = celeba_dist_nn(json_destination, c_id)

        a = Namespace(
            dataroot=f"{PROJ_ROOT}/celeba/img_align_celeba/",
            csv_file=f"{PROJ_ROOT}/celeba/list_attr_celeba.txt",
            delimiter_type="space",
            model=model_destination,
            gradcam_layer="layer4",
            batch_size=32,
            model_type="resnet",
            tree_json=json_destination,
            cluster_id=new_c_id,
            output_path=os.path.join(APP_ROOT, "images", output_file),
        )

        plot_celeba(a, l)

    session["output_file"] = output_file
    return redirect(url_for("display_cluster"))


@app.route("/display_cluster")
def display_cluster():
    return render_template("display_cluster.html", image_name=session["output_file"])


@app.route("/details", methods=["POST"])
def details():
    json_target = os.path.join(APP_ROOT, "files")

    m = request.form.get("model-name")
    acc = request.form.get("model-accuracy")

    j = request.form.get("json-name")
    json_destination = "/".join([json_target, j])

    d = request.form.get("dataset-select")

    t = request.form.get("threshold")

    if t == "":
        t = 2 / 3

    if d == "COCO":
        data, columns = convert_to_dict_coco(json_destination, acc, float(t))
        gc = request.form.get("gradcam-category")
    elif d == "CelebA":
        data, columns = convert_to_dict_celeba(json_destination, acc, float(t))
        gc = None

    return render_template(
        "table.html",
        data=data,
        columns=columns,
        title="Sorted Clusters:",
        model=m,
        tree=j,
        dataset=d,
        gradcam=gc,
    )


@app.route("/model_upload", methods=["POST"])
def model_upload():
    target = os.path.join(APP_ROOT, "models")

    if not os.path.isdir(target):
        os.mkdir(target)

    m = request.files.get("model")
    name = request.form.get("model-name")
    model_destination = "/".join([target, name + ".pth"])
    error = None
    success = None

    if not os.path.exists(model_destination):
        m.save(model_destination)
        success = "Saved model."
        print("Saved model.")
    else:
        error = "Model with the same name already exists."
        print("Model with the same name already exists.")

    return render_template("upload_model.html", success=success, error=error)


@app.route("/json_upload", methods=["POST"])
def json_upload():
    target = os.path.join(APP_ROOT, "files")

    if not os.path.isdir(target):
        os.mkdir(target)

    m = request.files.get("json")
    name = request.form.get("json-name")
    json_destination = "/".join([target, name + ".json"])
    error = None
    success = None

    if not os.path.exists(json_destination):
        m.save(json_destination)
        success = "Saved JSON."
        print("Saved JSON.")
    else:
        error = "JSON with the same name already exists."
        print("JSON with the same name already exists.")

    return render_template("upload_json.html", success=success, error=error)


@app.route("/upload/<filename>")
def send_image(filename):
    return send_from_directory("images", filename)
