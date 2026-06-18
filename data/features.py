import pandas as pd
import numpy as np


def feature_zoo(data):
    # x dataframe, [total, 6]
    # acc x, y, z; gyo x, y, z
    if data.shape[-1] == 6:
        raw_acc_x = data[:, :, 0]
        raw_acc_y = data[:, :, 1]
        raw_acc_z = data[:, :, 2]
        mean_acc_xyz = np.mean([raw_acc_x, raw_acc_y, raw_acc_z], axis=0)
        var_acc_xyz = np.var([raw_acc_x, raw_acc_y, raw_acc_z], axis=0)
        # std_acc_xyz = np.std([raw_acc_x, raw_acc_y, raw_acc_z], axis=0)
        norm_acc_1 = np.sum([np.abs(raw_acc_x), np.abs(raw_acc_y), np.abs(raw_acc_z)], axis=0)
        norm_acc_2 = np.sqrt(np.sum([raw_acc_x*raw_acc_x, raw_acc_y*raw_acc_y, raw_acc_z*raw_acc_z], axis=0))

        raw_gyo_x = data[:, :, 3]
        raw_gyo_y = data[:, :, 4]
        raw_gyo_z = data[:, :, 5]
        mean_gyo_xyz = np.mean([raw_gyo_x, raw_gyo_y, raw_gyo_z], axis=0)
        var_gyo_xyz = np.var([raw_gyo_x, raw_gyo_y, raw_gyo_z], axis=0)
        # std_gyo_xyz = np.std([raw_gyo_x, raw_gyo_y, raw_gyo_z], axis=0)
        norm_gyo_1 = np.sum([np.abs(raw_gyo_x), np.abs(raw_gyo_y), np.abs(raw_gyo_z)], axis=0)
        norm_gyo_2 = np.sqrt(np.sum([raw_gyo_x*raw_gyo_x, raw_gyo_y*raw_gyo_y, raw_gyo_z*raw_gyo_z], axis=0))

        data = np.concatenate([np.expand_dims(raw_acc_x, axis=2),
                               np.expand_dims(raw_acc_y, axis=2),
                               np.expand_dims(raw_acc_z, axis=2),
                               np.expand_dims(mean_acc_xyz, axis=2),
                               np.expand_dims(var_acc_xyz, axis=2),
                               # np.expand_dims(std_acc_xyz, axis=2),
                               np.expand_dims(norm_acc_1, axis=2),
                               np.expand_dims(norm_acc_2, axis=2),
                               np.expand_dims(raw_gyo_x, axis=2),
                               np.expand_dims(raw_gyo_y, axis=2),
                               np.expand_dims(raw_gyo_z, axis=2),
                               np.expand_dims(mean_gyo_xyz, axis=2),
                               np.expand_dims(var_gyo_xyz, axis=2),
                               # np.expand_dims(std_gyo_xyz, axis=2),
                               np.expand_dims(norm_gyo_1, axis=2),
                               np.expand_dims(norm_gyo_2, axis=2)], axis=2)
    else:
        raw_acc_x_thigh = data[:, :, 0]
        raw_acc_y_thigh = data[:, :, 1]
        raw_acc_z_thigh = data[:, :, 2]
        mean_acc_xyz_thigh = np.mean([raw_acc_x_thigh, raw_acc_y_thigh, raw_acc_z_thigh], axis=0)
        var_acc_xyz_thigh = np.var([raw_acc_x_thigh, raw_acc_y_thigh, raw_acc_z_thigh], axis=0)
        # std_acc_xyz_thigh = np.std([raw_acc_x_thigh, raw_acc_y_thigh, raw_acc_z_thigh], axis=0)
        norm_acc_1_thigh = np.sum([np.abs(raw_acc_x_thigh), np.abs(raw_acc_y_thigh), np.abs(raw_acc_z_thigh)], axis=0)
        norm_acc_2_thigh = np.sqrt(np.sum([raw_acc_x_thigh*raw_acc_x_thigh, raw_acc_y_thigh*raw_acc_y_thigh, raw_acc_z_thigh*raw_acc_z_thigh], axis=0))

        raw_gyo_x_thigh = data[:, :, 3]
        raw_gyo_y_thigh = data[:, :, 4]
        raw_gyo_z_thigh = data[:, :, 5]
        mean_gyo_xyz_thigh = np.mean([raw_gyo_x_thigh, raw_gyo_y_thigh, raw_gyo_z_thigh], axis=0)
        var_gyo_xyz_thigh = np.var([raw_gyo_x_thigh, raw_gyo_y_thigh, raw_gyo_z_thigh], axis=0)
        # std_gyo_xyz_thigh = np.std([raw_gyo_x_thigh, raw_gyo_y_thigh, raw_gyo_z_thigh], axis=0)
        norm_gyo_1_thigh = np.sum([np.abs(raw_gyo_x_thigh), np.abs(raw_gyo_y_thigh), np.abs(raw_gyo_z_thigh)], axis=0)
        norm_gyo_2_thigh = np.sqrt(np.sum([raw_gyo_x_thigh*raw_gyo_x_thigh, raw_gyo_y_thigh*raw_gyo_y_thigh, raw_gyo_z_thigh*raw_gyo_z_thigh], axis=0))

        raw_acc_x_shank = data[:, :, 6]
        raw_acc_y_shank = data[:, :, 7]
        raw_acc_z_shank = data[:, :, 8]
        mean_acc_xyz_shank = np.mean([raw_acc_x_shank, raw_acc_y_shank, raw_acc_z_shank], axis=0)
        var_acc_xyz_shank = np.var([raw_acc_x_shank, raw_acc_y_shank, raw_acc_z_shank], axis=0)
        # std_acc_xyz_shank = np.std([raw_acc_x_shank, raw_acc_y_shank, raw_acc_z_shank], axis=0)
        norm_acc_1_shank = np.sum([np.abs(raw_acc_x_shank), np.abs(raw_acc_y_shank), np.abs(raw_acc_z_shank)], axis=0)
        norm_acc_2_shank = np.sqrt(np.sum([raw_acc_x_shank*raw_acc_x_shank, raw_acc_y_shank*raw_acc_y_shank, raw_acc_z_shank*raw_acc_z_shank], axis=0))

        raw_gyo_x_shank = data[:, :, 9]
        raw_gyo_y_shank = data[:, :, 10]
        raw_gyo_z_shank = data[:, :, 11]
        mean_gyo_xyz_shank = np.mean([raw_gyo_x_shank, raw_gyo_y_shank, raw_gyo_z_shank], axis=0)
        var_gyo_xyz_shank = np.var([raw_gyo_x_shank, raw_gyo_y_shank, raw_gyo_z_shank], axis=0)
        # std_gyo_xyz_shank = np.std([raw_gyo_x_shank, raw_gyo_y_shank, raw_gyo_z_shank], axis=0)
        norm_gyo_1_shank = np.sum([np.abs(raw_gyo_x_shank), np.abs(raw_gyo_y_shank), np.abs(raw_gyo_z_shank)], axis=0)
        norm_gyo_2_shank = np.sqrt(np.sum([raw_gyo_x_shank*raw_gyo_x_shank, raw_gyo_y_shank*raw_gyo_y_shank, raw_gyo_z_shank*raw_gyo_z_shank], axis=0))

        raw_acc_x_foot = data[:, :, 12]
        raw_acc_y_foot = data[:, :, 13]
        raw_acc_z_foot = data[:, :, 14]
        mean_acc_xyz_foot = np.mean([raw_acc_x_foot, raw_acc_y_foot, raw_acc_z_foot], axis=0)
        var_acc_xyz_foot = np.var([raw_acc_x_foot, raw_acc_y_foot, raw_acc_z_foot], axis=0)
        # std_acc_xyz_foot = np.std([raw_acc_x_foot, raw_acc_y_foot, raw_acc_z_foot], axis=0)
        norm_acc_1_foot = np.sum([np.abs(raw_acc_x_foot), np.abs(raw_acc_y_foot), np.abs(raw_acc_z_foot)], axis=0)
        norm_acc_2_foot = np.sqrt(np.sum([raw_acc_x_foot*raw_acc_x_foot, raw_acc_y_foot*raw_acc_y_foot, raw_acc_z_foot*raw_acc_z_foot], axis=0))

        raw_gyo_x_foot = data[:, :, 15]
        raw_gyo_y_foot = data[:, :, 16]
        raw_gyo_z_foot = data[:, :, 17]
        mean_gyo_xyz_foot = np.mean([raw_gyo_x_foot, raw_gyo_y_foot, raw_gyo_z_foot], axis=0)
        var_gyo_xyz_foot = np.var([raw_gyo_x_foot, raw_gyo_y_foot, raw_gyo_z_foot], axis=0)
        # std_gyo_xyz_foot = np.std([raw_gyo_x_foot, raw_gyo_y_foot, raw_gyo_z_foot], axis=0)
        norm_gyo_1_foot = np.sum([np.abs(raw_gyo_x_foot), np.abs(raw_gyo_y_foot), np.abs(raw_gyo_z_foot)], axis=0)
        norm_gyo_2_foot = np.sqrt(np.sum([raw_gyo_x_foot*raw_gyo_x_foot, raw_gyo_y_foot*raw_gyo_y_foot, raw_gyo_z_foot*raw_gyo_z_foot], axis=0))

        data = np.concatenate([np.expand_dims(raw_acc_x_thigh, axis=2),
                               np.expand_dims(raw_acc_y_thigh, axis=2),
                               np.expand_dims(raw_acc_z_thigh, axis=2),
                               np.expand_dims(mean_acc_xyz_thigh, axis=2),
                               np.expand_dims(var_acc_xyz_thigh, axis=2),
                               # np.expand_dims(std_acc_xyz_thigh, axis=2),
                               np.expand_dims(norm_acc_1_thigh, axis=2),
                               np.expand_dims(norm_acc_2_thigh, axis=2),
                               np.expand_dims(raw_gyo_x_thigh, axis=2),
                               np.expand_dims(raw_gyo_y_thigh, axis=2),
                               np.expand_dims(raw_gyo_z_thigh, axis=2),
                               np.expand_dims(mean_gyo_xyz_thigh, axis=2),
                               np.expand_dims(var_gyo_xyz_thigh, axis=2),
                               # np.expand_dims(std_gyo_xyz_thigh, axis=2),
                               np.expand_dims(norm_gyo_1_thigh, axis=2),
                               np.expand_dims(norm_gyo_2_thigh, axis=2),

                               np.expand_dims(raw_acc_x_shank, axis=2),
                               np.expand_dims(raw_acc_y_shank, axis=2),
                               np.expand_dims(raw_acc_z_shank, axis=2),
                               np.expand_dims(mean_acc_xyz_shank, axis=2),
                               np.expand_dims(var_acc_xyz_shank, axis=2),
                               # np.expand_dims(std_acc_xyz_shank, axis=2),
                               np.expand_dims(norm_acc_1_shank, axis=2),
                               np.expand_dims(norm_acc_2_shank, axis=2),
                               np.expand_dims(raw_gyo_x_shank, axis=2),
                               np.expand_dims(raw_gyo_y_shank, axis=2),
                               np.expand_dims(raw_gyo_z_shank, axis=2),
                               np.expand_dims(mean_gyo_xyz_shank, axis=2),
                               np.expand_dims(var_gyo_xyz_shank, axis=2),
                               # np.expand_dims(std_gyo_xyz_shank, axis=2),
                               np.expand_dims(norm_gyo_1_shank, axis=2),
                               np.expand_dims(norm_gyo_2_shank, axis=2),

                               np.expand_dims(raw_acc_x_foot, axis=2),
                               np.expand_dims(raw_acc_y_foot, axis=2),
                               np.expand_dims(raw_acc_z_foot, axis=2),
                               np.expand_dims(mean_acc_xyz_foot, axis=2),
                               np.expand_dims(var_acc_xyz_foot, axis=2),
                               # np.expand_dims(std_acc_xyz_foot, axis=2),
                               np.expand_dims(norm_acc_1_foot, axis=2),
                               np.expand_dims(norm_acc_2_foot, axis=2),
                               np.expand_dims(raw_gyo_x_foot, axis=2),
                               np.expand_dims(raw_gyo_y_foot, axis=2),
                               np.expand_dims(raw_gyo_z_foot, axis=2),
                               np.expand_dims(mean_gyo_xyz_foot, axis=2),
                               np.expand_dims(var_gyo_xyz_foot, axis=2),
                               # np.expand_dims(std_gyo_xyz_foot, axis=2),
                               np.expand_dims(norm_gyo_1_foot, axis=2),
                               np.expand_dims(norm_gyo_2_foot, axis=2)], axis=2)
    return data