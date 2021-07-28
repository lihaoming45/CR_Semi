import numpy as np
import SimpleITK as sitk
import glob
import nibabel as nib


def itk_resample(itk_image, new_spacing=None, new_shape=None, islabel=False):
    '''
    itk_image:
    new_spacing: list or tuple, [x,y,z]
    new_shape: list or tuple, [x,y]
    islabel: if True, using Interpolator `sitk.sitkNearestNeighbor`
    '''
    size = np.array(itk_image.GetSize())
    spacing = np.array(itk_image.GetSpacing())

    if new_shape is not None:
        assert new_spacing is None

    if new_spacing is not None:
        assert new_shape is None

    # if (spacing == new_spacing).all() and new_shape is None:
    #     return itk_image
    # if (size[:-1] == list(new_shape)).all() and new_spacing is None:
    #     return itk_image

    if new_spacing is not None:
        new_spacing = np.array(new_spacing)
        new_size = size * spacing / new_spacing
        new_spacing_refine = size * spacing / new_size
        new_spacing_refine = [float(s) for s in new_spacing_refine]
        new_size = [int(round(s)) for s in new_size]

    if new_shape is not None:
        new_shape = np.array(new_shape)
        if len(new_shape)==2:
            ratio = np.mean(new_shape / size[:-1])
            new_size = np.concatenate([new_shape, [int(ratio * size[-1])]])
        elif len(new_shape)==3:
            new_size = np.concatenate([new_shape[-2:],[new_shape[0]]])
        else:
            raise ImportError("new_shape维度必须等于2或3")
        new_spacing = size * spacing / new_size
        new_spacing_refine = new_spacing
        new_spacing_refine = [float(s) for s in new_spacing_refine]
        new_size = [int(round(s)) for s in new_size]

    #new_size = (288,256,224)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)

    if islabel:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetInterpolator(sitk.sitkLinear)

    new_itkimage = resample.Execute(itk_image)
    return new_itkimage


if __name__ == "__main__":
    label_path = "H:/formal_data_3D_GE/all_data_Decode/label/20190428-52103-05-03_L_I0000012_label.nii.gz"
    data_path = "H:\\formal_data_3D_GE\\all_data_Decode\data\\20190428-52103-05-03_L_I0000012.nii.gz"
    save_path = "H:\\formal_data_3D_GE\\"

    label_name = label_path.split('/')[-1]
    data_name = data_path.split('\\')[-1]
    # label_vol = nib.load(label_path).get_data()
    label_vol =sitk.ReadImage(label_path)
    data_vol = sitk.ReadImage(data_path)

    out_data = itk_resample(data_vol, new_shape=(288, 256), islabel=False)
    out_data = sitk.GetArrayFromImage(out_data)

    out_label = itk_resample(label_vol, new_shape=(288, 256), islabel=True)
    out_label = sitk.GetArrayFromImage(out_label)
    print(out_label.shape)

