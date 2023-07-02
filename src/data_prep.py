
def get_paths_and_labels(csv, source_path):

    binary_class = []
    filepaths = []
    for key, value in csv.iterrows():
        binary_class.append(value[1])
        subject_path = source_path + 'fmri_' + value[0][:-1] + '_session1_run1'
        fmri_path = subject_path + '.nii.gz'
        filepaths.append(fmri_path)
    return filepaths, binary_class