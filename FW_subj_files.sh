#!/bin/sh
python unstr_gen_energy_label_inall.py 2 'inlabStr'
python unstr_split_trainset_31hz.py 2 'inlabStr'
python unstr_engyLabel2rawdataLabel.py 2 'inlabStr'