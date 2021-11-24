# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:41:35 2020

@author: sandr
"""
import time
import csv
import os.path

""" Write the results in CSV """

def write_results(file_to_save, start_time,model_name,time_window,overlap, epoch, batch_size,acc,std_acc,recall,f1_score, unlabeled_percetage):

    total_computing_time = time.time() - start_time
    print("computing time:", str(total_computing_time))
    # La 'a' es para que no nos sobreescriba el documento, es decir, inserte las nuevas filas abajo

    filename = file_to_save
    
    file_exists = os.path.isfile(filename)  # Compruebo si existe el fichero

    with open(filename, 'a') as csvfile:
        headers = ['Model','TimeWindow','Overlap','Epoch','BatchSize','TimeComputing','Accuracy','Loss','recall','F1-Acore','unlabeled_data(percentage)']
        writer = csv.DictWriter(csvfile, fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # El archivo aún no existe por lo que me lo creo con la cabecera
        writer.writerow({'Model': model_name,'TimeWindow': time_window, 'Overlap': overlap,
                         'Epoch': epoch,'BatchSize':batch_size,'TimeComputing': total_computing_time,
                         'Accuracy':acc,'Loss':std_acc,'recall':recall,'F1-Acore':f1_score,
                         'unlabeled_data(percentage)':unlabeled_percetage})