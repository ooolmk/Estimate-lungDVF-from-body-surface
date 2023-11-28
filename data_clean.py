import os

import numpy as np
import pydicom


def get_3dct(PathDicom):
    # 通过文件夹整合3dct
    FilesDCM = []
    # 将所有dicom文件读入
    for filename in os.listdir(PathDicom):
        if ".dcm" in filename.lower():  # 判断文件是否为dicom文件
            FilesDCM.append(pydicom.dcmread(os.path.join(PathDicom, filename)))
    img_shape = list(FilesDCM[0].pixel_array.shape)
    img_shape.append(len(FilesDCM))
    img3d = np.zeros(img_shape)
    # 遍历所有的dicom文件，读取图像数据，存放在numpy数组中
    for i in range(len(FilesDCM)):
        img2d = FilesDCM[i].pixel_array
        img3d[:, :, i] = img2d
    return img3d


def get_phase_path(path='./4D-Lung'):
    patient_list = []
    time_list = []
    time_dict = {}

    numpy_path = './4D-Lung-numpy0'
    numpy_patient_list = []
    numpy_time_list = []
    numpy_time_dict = {}

    for patient in os.listdir(path):
        patient_dir = os.path.join(path, patient)
        patient_list.append(patient_dir)

        numpy_patient_dir = os.path.join(numpy_path, patient)
        numpy_patient_list.append(numpy_patient_dir)

        for time in os.listdir(patient_dir):
            time_dir = os.path.join(patient_dir, time)
            phase_list = []

            numpy_time_dir = os.path.join(numpy_patient_dir, time)
            numpy_phase_list = []

            for phase in os.listdir(time_dir):
                phase_dir = os.path.join(time_dir, phase)
                numpy_phase_dir = os.path.join(numpy_time_dir, phase)

                if phase.count('.') == 2:
                    phase_list.append(phase_dir)
                    numpy_phase_list.append(numpy_phase_dir)

            if len(phase_list) > 1:
                time_list.append(time_dir)
                time_dict[time_dir] = phase_list

                numpy_time_list.append(numpy_time_dir)
                numpy_time_dict[numpy_time_dir] = numpy_phase_list
    # print(patient_list, time_list, len(phase_list))

    return patient_list, time_list, time_dict, numpy_patient_list, numpy_time_list, numpy_time_dict

# remove 4DCBCT use 4DFBCT only
patient_list, time_list, time_dict, numpy_patient_list, numpy_time_list, numpy_time_dict = get_phase_path()
for id0, time_path in enumerate(time_list):
    numpy_time_path = numpy_time_list[id0]
    os.makedirs(numpy_time_path)
    for id1, phase_path in enumerate(time_dict[time_path]):
        numpy_phase_path = numpy_time_dict[numpy_time_path][id1]
        # print(numpy_phase_path)
        # print(phase_path)
        try:
            ct = get_3dct(phase_path)
        except:
            print(f'error {phase_path}')
            continue
        if ct.min() > -1900:
            np.save(numpy_phase_path, ct)
            print(ct.shape, ct.min())
        # else:
        #     np.save(numpy_phase_path, ct)
