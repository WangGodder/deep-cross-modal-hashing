# coding: utf-8
# author: Godder
# Date: 2021/5/11
# Github: https://github.com/WangGodder

import os
import datetime
from string import Template

abs_dir = os.path.abspath(os.path.dirname(__file__))

trainer_tpl_path = os.path.join(abs_dir, "training", "training.tpl")
trainer_out_path = os.path.join(abs_dir, "training")


def create_train(train_name: str):
    create_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    tpl_file = open(trainer_tpl_path)
    file_name = os.path.join(trainer_out_path, train_name+".py")
    trainer_file = open(file_name, "w")

    lines = []
    tpl = Template(tpl_file.read())

    lines.append(tpl.substitute(create_time=create_time, train_name=train_name))

    trainer_file.writelines(lines)
    trainer_file.close()
    tpl_file.close()
    print("create trainer %s finish" % train_name)

