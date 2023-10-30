import nibabel as nib
import numpy as np
from netneurotools import datasets as nntdata
from neuromaps.images import annot_to_gifti

DATADIR = "mats"


class Atlas:
    def __init__(self, name: str, label_img: np.ndarray, labels: dict):
        self.name = name
        self.label_img = label_img
        self.id2label = labels
        self.label2id = {v: k for k, v in labels.items()}

    def label(self, key: int) -> str:
        return self.id2label.get(key)

    def key(self, label: str) -> int:
        return self.label2id.get(label)

    def __getitem__(self, key) -> str | int:
        if isinstance(key, int):
            return self.label(key)
        elif isinstance(key, str):
            return self.key(key)
        else:
            raise ValueError("key is incorrect type")

    def __len__(self) -> int:
        return len(self.id2label) - 1

    def vox_to_parc(
        self, values: np.ndarray, agg_func=np.mean, axis: int = -1
    ) -> np.ndarray:
        n_parcels = len(self)
        parcellation = self.label_img

        if values.ndim == 1:
            values = values[np.newaxis, :]

        new_shape = list(values.shape)
        new_shape[axis] = n_parcels
        parcel_values = np.zeros(new_shape, dtype=values.dtype)
        for i in range(1, n_parcels + 1):
            parcel_mask = parcellation == i
            parcel_values[:, i - 1] = agg_func(values[:, parcel_mask], axis=axis)
        return parcel_values

    def parc_to_vox(self, values: np.ndarray) -> np.ndarray:
        parcellation = self.label_img
        voxel_values = np.zeros_like(self.label_img, dtype=values.dtype)
        for i in range(1, len(self)):
            parcel_mask = parcellation == i
            voxel_values[parcel_mask] = values[0, i]  # NOTE

        return voxel_values

    def parcellate(self, values: np.ndarray, **kwargs) -> np.ndarray:
        return self.parc_to_vox(self.vox_to_parc(values, **kwargs))

    def get_background_mask(self) -> np.ndarray:
        return self.label_img == 0

    @staticmethod
    def schaefer2018(rois: int = 1000, networks: int = 17):
        filenames = nntdata.fetch_schaefer2018(version="fsaverage6", data_dir=DATADIR)
        atlasname = f"{rois}Parcels{networks}Networks"
        atlas_bunch = filenames[atlasname]
        gLh, gRh = annot_to_gifti((atlas_bunch.lh, atlas_bunch.rh))
        label_img = np.concatenate((gLh.agg_data(), gRh.agg_data()))
        labels = (
            gLh.labeltable.get_labels_as_dict() | gRh.labeltable.get_labels_as_dict()
        )
        return Atlas(atlasname, label_img, labels)

    @staticmethod
    def glasser2016():
        atlasname = "glasser2016"
        gLh = nib.load(f"{DATADIR}/tpl-fsaverage6_hemi-L_desc-MMP_dseg.label.gii")
        gRh = nib.load(f"{DATADIR}/tpl-fsaverage6_hemi-R_desc-MMP_dseg.label.gii")
        label_img = np.concatenate((gLh.agg_data(), gRh.agg_data()))
        labels = (
            gLh.labeltable.get_labels_as_dict() | gRh.labeltable.get_labels_as_dict()
        )
        return Atlas(atlasname, label_img, labels)
