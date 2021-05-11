# coding: utf-8
# author: Godder
# Date: 2021/5/11
# Github: https://github.com/WangGodder

import os
import datetime
from string import Template

__all__ = ["create_new_algorithm"]

abs_dir = os.path.abspath(os.path.dirname(__file__))

trainer_tpl_path = os.path.join(abs_dir, "utils", "templates", "training.tpl")
model_init_tpl_path = os.path.join(abs_dir, "utils", "templates", "model_init.tpl")
trainer_out_path = os.path.join(abs_dir, "training")
model_out_path = os.path.join(abs_dir, "models")


def create_model(train_name: str):
    create_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    tpl_file = open(model_init_tpl_path)
    dir_name = os.path.join(model_out_path, train_name)
    if os.path.exists(dir_name):
        raise FileExistsError("model dir %s exist, create fail" % train_name)
    os.mkdir(dir_name, 0o777)
    model_init_file_name = os.path.join(dir_name, "__init__.py")
    model_init_file = open(model_init_file_name, "w")

    lines = []
    tpl = Template(tpl_file.read())
    lines.append(tpl.substitute(create_time=create_time))
    model_init_file.writelines(lines)
    model_init_file.close()
    tpl_file.close()


def create_train(train_name: str):
    create_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    tpl_file = open(trainer_tpl_path)
    file_name = os.path.join(trainer_out_path, train_name+".py")
    if os.path.exists(file_name):
        raise FileExistsError("train file %s exist, create fail" % train_name)
    trainer_file = open(file_name, "w")

    lines = []
    tpl = Template(tpl_file.read())
    lines.append(tpl.substitute(create_time=create_time, train_name=train_name))
    trainer_file.writelines(lines)
    trainer_file.close()
    tpl_file.close()


def create_new_algorithm(train_name: str):
    create_train(train_name)
    create_model(train_name)
    print("create trainer %s finish" % train_name)

