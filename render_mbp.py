"""Camera.

Concepts:
    - Create and mount cameras
    - Render RGB images, point clouds, segmentation masks
"""

import sapien
import numpy as np
from PIL import Image, ImageColor

import trimesh
import os
import json

def cal_mat44(cam_pos):
    # Compute the camera pose by specifying forward(x), left(y) and up(z)
    # The camera is looking at the origin
    forward = -cam_pos / np.linalg.norm(cam_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos

    return mat44


def main():
    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    urdf_path = "../src/mbp/mobility.urdf"
    object = loader.load(urdf_path)
    assert object, "failed to load URDF."
    
    # Only one joint for this object
    joint_limits = object.get_qlimits()
    q_low, q_high = joint_limits[0, 0], joint_limits[0, 1]

    # object.set_qpos(np.array([[-1.]]))

    # Light sources
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1])
    scene.add_point_light([1, -2, 2], [1, 1, 1])
    scene.add_point_light([-1, 0, 1], [1, 1, 1])

    # Camera
    near, far = 0.1, 100
    width, height = 1280, 960

    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=np.deg2rad(35),
        near=near,
        far=far,
    )

    init_cam_pos = np.array([-2, -2, 3])

    num_cam = 24
    num_frames = 10

    train_ids = [i for i in range(num_cam) if i % 6 != 0]
    test_ids = [i for i in range(num_cam) if i % 6 == 0]

    K = camera.get_intrinsic_matrix()
    test_k = np.array(K)[None, None, ...].repeat(num_frames, axis=0).repeat(len(test_ids), axis=1).tolist()
    train_k = np.array(K)[None, None, ...].repeat(num_frames, axis=0).repeat(len(train_ids), axis=1).tolist()
    test_fn = [['{}/{:06}.jpg'.format(cam, t) for cam in test_ids] for t in range(num_frames)]
    train_fn = [['{}/{:06}.jpg'.format(cam, t) for cam in train_ids] for t in range(num_frames)]

    train_json = {'w': width, 'h': height, 'k': train_k, 'fn': train_fn, 'cam_id': np.array(train_ids)[None, ...].repeat(num_frames, axis=0).tolist()}
    test_json = {'w': width, 'h': height, 'k': test_k, 'fn': test_fn, 'cam_id': np.array(test_ids)[None, ...].repeat(num_frames, axis=0).tolist()}

    train_w2cs = []
    test_w2cs = []

    for cam_i, angle in enumerate(np.linspace(0, 2 * np.pi, num_cam)):
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
        cam_pos = rotation @ init_cam_pos
        mat44 = cal_mat44(cam_pos)

        camera.entity.set_pose(sapien.Pose(mat44))

        save_dir = '../data/mbp/ims/{}'.format(cam_i)
        articulate_label_dir = '../data/mbp/art_labels/{}'.format(cam_i)
        seg_dir = '../data/mbp/seg/{}'.format(cam_i)

        # TODO: Add camera matrix output
        scene.update_render()
        c2w = np.array(camera.get_model_matrix())
        w2c = np.linalg.inv(c2w)
        # w2c[0, :] *= -1
        w2c[1, :] *= -1
        w2c[2, :] *= -1

        if cam_i in test_ids:
            test_w2cs.append(w2c.tolist())
        else:
            train_w2cs.append(w2c.tolist())

        # for ratio_i, ratio in enumerate(np.linspace(1, 0, num_frames)):
        #     qpos = q_low + ratio * (q_high - q_low)
        #     object.set_qpos(np.array([[qpos]]))

        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     if not os.path.exists(articulate_label_dir):
        #         os.makedirs(articulate_label_dir)
        #     if not os.path.exists(seg_dir):
        #         os.makedirs(seg_dir)

        #     # scene.step()  # run a physical step
        #     scene.update_render()  # sync pose from SAPIEN to renderer
        #     camera.take_picture()  # submit rendering jobs to the GPU

        #     # ---------------------------------------------------------------------------- #
        #     # RGBA
        #     # ---------------------------------------------------------------------------- #
        #     rgba = camera.get_picture("Color")  # [H, W, 4]
        #     rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        #     # Black background
        #     rgb_img = rgba_img[..., :3] * (rgba_img[..., 3:] > 0)
        #     rgb_pil = Image.fromarray(rgb_img)
        #     rgb_pil.save(os.path.join(save_dir, '{:06}.jpg'.format(ratio_i)))

        #     # ---------------------------------------------------------------------------- #
        #     # XYZ position in the camera space
        #     # ---------------------------------------------------------------------------- #
        #     # Each pixel is (x, y, z, render_depth) in camera space (OpenGL/Blender)
        #     position = camera.get_picture("Position")  # [H, W, 4]

        #     # OpenGL/Blender: y up and -z forward
        #     points_opengl = position[..., :3][position[..., 3] < 1]
        #     points_color = rgba[position[..., 3] < 1]
        #     # Model matrix is the transformation from OpenGL camera space to SAPIEN world space
        #     # camera.get_model_matrix() must be called after scene.update_render()!
        #     model_matrix = camera.get_model_matrix()
        #     points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]

        #     points_color = (np.clip(points_color, 0, 1) * 255).astype(np.uint8)
        #     # trimesh.PointCloud(points_world, points_color).show()

        #     # ---------------------------------------------------------------------------- #
        #     # Segmentation labels
        #     # ---------------------------------------------------------------------------- #
        #     # Each pixel is (visual_id, actor_id/link_id, 0, 0)
        #     # visual_id is the unique id of each visual shape
        #     seg_labels = camera.get_picture("Segmentation")  # [H, W, 4]
        #     colormap = sorted(set(ImageColor.colormap.values()))
        #     color_palette = np.array(
        #         [ImageColor.getrgb(color) for color in colormap], dtype=np.uint8
        #     )
        #     # label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
        #     label1_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
        #     # Or you can use aliases below
        #     # label0_image = camera.get_visual_segmentation()
        #     # label1_image = camera.get_actor_segmentation()
        #     # label0_pil = Image.fromarray(color_palette[label0_image])
        #     # label0_pil.save("label0.png")
        #     label1_pil = Image.fromarray(color_palette[label1_image])
        #     label1_pil.save(os.path.join(articulate_label_dir, '{:06}.png'.format(ratio_i)))

        #     seg = rgba_img[..., -1].astype(bool)
        #     Image.fromarray(seg).save(os.path.join(seg_dir, '{:06}.png'.format(ratio_i)))

    train_json['w2c'] = np.array(train_w2cs)[None, ...].repeat(num_frames, axis=0).tolist()
    test_json['w2c'] = np.array(test_w2cs)[None, ...].repeat(num_frames, axis=0).tolist()

    with open('../data/mbp/train_meta.json', 'w') as f:
        json.dump(train_json, f)

    with open('../data/mbp/test_meta.json', 'w') as f:
        json.dump(test_json, f)

if __name__ == "__main__":
    main()
