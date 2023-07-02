from nilearn.connectome import ConnectivityMeasure
from nilearn import decomposition, image, plotting, input_data
import csv


def group_ica(fmri_files):
    ica = decomposition.CanICA(n_components=20, mask_strategy='whole-brain-template')
    print('decomposition done')
    ica.fit(fmri_files)
    print('fitting done')
    components = ica.components_
    components_inverse = ica.masker_.inverse_transform(components)
    # plotting.plot_stat_map(image.index_img(components_inverse, 10))
    # plotting.show()

    # plotting.plot_prob_atlas(components_inverse)
    # plotting.show()

    masker = input_data.NiftiMapsMasker(components_inverse,
                                        smoothing_fwhm=6, detrend=True,
                                        t_r=2.5, low_pass=0.1, high_pass=0.01)
    time_series_subjects = []
    i = 1
    for f in fmri_files:
        print(i, 'Currently processing:', f)
        i += 1
        time_series = masker.fit_transform(f)
        time_series_subjects.append(time_series)

    print('time_series_subjects array filled')
    correlation_measure = ConnectivityMeasure(kind='correlation')
    corr_mats = correlation_measure.fit_transform(time_series_subjects)

    # coords = plotting.find_probabilistic_atlas_cut_coords(components_inverse)
    # plotting.plot_connectome(corr_mats[0], coords, edge_threshold="80%")
    # plotting.show()

    return corr_mats


def write_to_csv(matrices, csv_path):
    print("The datatype of the matrices we're workikng with:")
    print(type(matrices[0]))
    with open(csv_path, 'w') as f:
        for m in matrices:
            one_dim = m.reshape(1, -1).tolist()
            writer = csv.writer(f)
            writer.writerow(one_dim)