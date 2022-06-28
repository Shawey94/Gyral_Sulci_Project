#add temporal patterns
#automatic adjust source image in index.html
from distutils import filelist
from fileinput import filename
from flask import Flask, render_template, url_for
import os
import argparse
import shutil
import chardet

figs_path = '/media/shawey/SSD8T/GyraSulci_Motor/Twin_Transformers/VisualizationInBroswer/temporal_pngs'
html_path = '/media/shawey/SSD8T/GyraSulci_Motor/Twin_Transformers/VisualizationInBroswer/templates'

image_names = os.listdir(path = figs_path)
record_id = []
#211215_14_gyri_sulci.png
for image in image_names:
    fig_id = image.split('_')[0]
    if (fig_id not in record_id):
        record_id.append(fig_id)

print('sub ids are ',record_id)

for sub_id in record_id:
    print(sub_id)
    shutil.copyfile(html_path + '/index.html', html_path+'/'+ str(sub_id)+'.html')

    with open(html_path + '/index.html', 'r', encoding='utf-8') as f_index: #, encoding='utf-8'
        with open(html_path+'/'+ str(sub_id)+'.html', 'w+', encoding='utf-8') as f: #, encoding='utf-8'
            contents = f_index.read()
            contents = contents.replace('111111',sub_id)
            print(contents)
            f.write(contents)




