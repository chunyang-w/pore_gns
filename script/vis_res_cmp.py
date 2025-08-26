import pyvista as pv
import numpy as np
from pore_net.utils import cube2mesh

# Set up device


save_fig = False
num_frames = 29
move_camera = False
show_error = True


# Input channel 1
# cube_pred = np.load("/Users/chunyang/Downloads/pred (4).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (5).npy")

# input channel 5
# cube_pred = np.load("/Users/chunyang/Downloads/pred (1).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (2).npy")

# with pore wall input channel 5 BCE
# cube_pred = np.load("/Users/chunyang/Downloads/pred (11).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (12).npy")

# with pore wall input channel 5 BCE - auto-reg on 150-200
# cube_pred = np.load("/Users/chunyang/Downloads/pred (15).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (16).npy")

# with pore wall input channel 5 BCE - auto-reg on 50-100
# cube_pred = np.load("/Users/chunyang/Downloads/pred (16).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (17).npy")

# with pore wall input channel 5 BCE - no auto-reg on 50-100
# cube_pred = np.load("/Users/chunyang/Downloads/pred (17).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (18).npy")

# # with pore wall input channel 5 BCE - auto-reg on 150-200, but with step = 1
# cube_pred = np.load("/Users/chunyang/Downloads/pred (18).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (19).npy")

# # with pore wall input channel 5 BCE - pooling on 148-197 1 to 1
# cube_pred = np.load("/Users/chunyang/Downloads/pred (20).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (21).npy")

# # with pore wall input channel 5 BCE - pooling on 1 to 1 test on 072 first 50 frames - 17.8% error
# cube_pred = np.load("/Users/chunyang/Downloads/pred (22).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (23).npy")

# # with pore wall input channel 5 BCE - pooling on 1 to 1 test on 073 150-179
# cube_pred = np.load("/Users/chunyang/Downloads/pred (24).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (25).npy")

# # with pore wall input channel 5 BCE - batch size alot, fp16 to 1 test on 073 150-179
# cube_pred = np.load("/Users/chunyang/Downloads/pred (3).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (3).npy")

# # with pore wall input channel 5 BCE - batch size 3, fp16 to 1 test on 073 150-179, with normalvel
# cube_pred = np.load("/Users/chunyang/Downloads/pred (4).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (4).npy")

# with pore wall input channel 5 BCE - batch size 3, fp16 to 1 test on 072 110-139 with normalvel
# cube_pred = np.load("/Users/chunyang/Downloads/pred (7).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (7).npy")

# # Input channel 1, no_phy 150-200
cube_pred = np.load("/Users/chunyang/Downloads/pred (6).npy")
cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (6).npy")


# 2 previous frames, 1 ouput fp16
# cube_pred = np.load("/Users/chunyang/Downloads/pred (8).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (8).npy")


# 2 previous frames, 1 ouput fp16, on 072
# frame 150-180
# cube_pred = np.load("/Users/chunyang/Downloads/pred (32).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (34).npy")
# # frame 30-90
# cube_pred = np.load("/Users/chunyang/Downloads/pred (33).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (26).npy")
# # frame 60-110
# cube_pred = np.load("/Users/chunyang/Downloads/pred (34).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (27).npy")

# # # 3 previous frames, 1 ouput fp16 on 072
# cube_pred = np.load("/Users/chunyang/Downloads/pred (29).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (31).npy")

# # # 5 previous frames, 1 ouput fp16 on 072
# # frame 60-110
# cube_pred = np.load("/Users/chunyang/Downloads/pred (35).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (28).npy")


# # # 5 previous frames, 1 ouput fp16 on 073
# # frame 150-195
# cube_pred = np.load("/Users/chunyang/Downloads/pred (36).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (29).npy")

# # 2 previous frames, 1 ouput fp16 on 073 - 150-194 frames
# cube_pred = np.load("/Users/chunyang/Downloads/pred (37).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (30).npy")


# # 2 previous frames, 1 ouput fp16 on 073 - 150-180 frames - no roll out
# cube_pred = np.load("/Users/chunyang/Downloads/pred (38).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (31).npy")

# # 2 previous frames, 1 ouput fp16 on 073 - 150-180 frames - yes roll out
# cube_pred = np.load("/Users/chunyang/Downloads/pred (40).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (35).npy")

# # 2 previous frames, 1 ouput fp16 on 073 - 100-130 frames - no roll out
# cube_pred = np.load("/Users/chunyang/Downloads/pred (39).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (33).npy")


# # 2 previous frames, 1 ouput fp16 on 073 - 100-130 frames - yes roll out
# cube_pred = np.load("/Users/chunyang/Downloads/pred (41).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (36).npy")


# # # 2 previous frames, 1 ouput fp16 on 073 - 30-60 frames - no roll out
# # cube_pred = np.load("/Users/chunyang/Downloads/pred (42).npy")
# # cube_gt = np.load("/Users/chunyang/Downloads/cube_gt (37).npy")

# 2 prev frames, 1 out, cross modality version, on 073, 150-180 frames ** good one
# cube_pred = np.load("/Users/chunyang/Downloads/73_t150-180_pd.npy")
# cube_gt = np.load("/Users/chunyang/Downloads/73_t150-180_gt.npy")


# # 2 prev frames, 1 out, cross modality version, on 073, 150-180 frames ** bad one, with -1 into 32 patch
# cube_pred = np.load("/Users/chunyang/Downloads/73_t150-180_pd (1).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/73_t150-180_gt (1).npy")


# # 2 prev frames, 1 out, cross modality version, on 072, 150-180 frames
# cube_pred = np.load("/Users/chunyang/Downloads/72_t150-180_pd.npy")
# cube_gt = np.load("/Users/chunyang/Downloads/72_t150-180_gt.npy")


# # 2 prev frames, 1 out, cross modality version, on 072, 120-190 frames
# cube_pred = np.load("/Users/chunyang/Downloads/73_t120-190_pd.npy")
# cube_gt = np.load("/Users/chunyang/Downloads/73_t120-190_gt.npy")

# # # 2 prev frames, 1 out, cross modality version, on 073, 150-180 frames ** no-reg
# cube_pred = np.load("/Users/chunyang/Downloads/73_t150-180_pd (3).npy")
# cube_gt = np.load("/Users/chunyang/Downloads/73_t150-180_gt (3).npy")


def calculate_dice_score(cube_pred, cube_gt, threshold=0.5):
    # Binarize the cubes using a threshold (e.g., 0.5) to make them binary (0 or 1)
    cube_pred_bin = (cube_pred > threshold).astype(np.int32)
    cube_gt_bin = (cube_gt > threshold).astype(np.int32)

    # Compute the intersection (common 1s) and union
    intersection = np.sum(cube_pred_bin * cube_gt_bin)
    union = np.sum(cube_pred_bin) + np.sum(cube_gt_bin)

    # Compute Dice score
    dice_score = (
        2 * intersection / (union + 1e-8)
    )  # Add epsilon to avoid division by zero

    return dice_score


num_frames = cube_pred.shape[0]
# Calculate Dice scores for each frame
dice_list = []
for i in range(num_frames):
    dice_list.append(calculate_dice_score(cube_pred[i], cube_gt[i]))
# Compute the overall average Dice score
dice_mean = np.mean(np.array(dice_list))
print(f"Overall Dice Score: {dice_mean:.4f}")


t = 0
mesh_pred = cube2mesh(cube_pred[t], threshold=0, relaxation_factor=0.2, smooth_iter=20)
mesh_gt = cube2mesh(cube_gt[t], threshold=0, relaxation_factor=0.2, smooth_iter=20)


# Setup side-by-side plotting
plotter = pv.Plotter(shape=(1, 2), window_size=(1800, 900))

# Left plot: mesh_pred
plotter.subplot(0, 0)
plotter.add_mesh(mesh_pred, color="lightblue")
plotter.add_text("Predicted Mesh", font_size=20)
text_error_actor = None
if show_error:
    text_error_actor = plotter.add_text(
        f"Overall Dice Score: {dice_mean:.4f}",
        position="lower_left",
        font_size=20,
    )
plotter.add_axes()

# Right plot: mesh_gt
plotter.subplot(0, 1)
plotter.add_mesh(mesh_gt, color="lightblue")
plotter.add_text("Ground Truth Mesh", font_size=20)
plotter.add_axes()

# Link camera views
plotter.link_views()


def update_slice(t):
    global text_error_actor
    t = int(t)
    mesh_pred_t1 = cube2mesh(
        cube_pred[t],
        threshold=0,
        relaxation_factor=0.2,
        smooth_iter=20,
    )
    mesh_pred.points = mesh_pred_t1.points
    mesh_pred.faces = mesh_pred_t1.faces
    mesh_gt_t1 = cube2mesh(
        cube_gt[t], threshold=0, relaxation_factor=0.2, smooth_iter=20
    )
    mesh_gt.points = mesh_gt_t1.points
    mesh_gt.faces = mesh_gt_t1.faces
    print("showing frame t: ", t)


plotter.camera_position = "zx"
plotter.camera.azimuth = 80
plotter.camera.elevation = 15

if not save_fig:
    plotter.add_slider_widget(
        update_slice,
        [0, num_frames],
        value=0,
        title="Slice",
    )
    plotter.show()
elif save_fig:
    plotter.open_gif(f"compare_move_camera_{move_camera}.gif", fps=8)
    text_actor = plotter.add_text(
        "Frame: 0", position="upper_right", font_size=20
    )  # noqa E501
    plotter.camera.zoom(1.2)
    for i in range(num_frames):
        plotter.remove_actor(text_actor)
        text_actor = plotter.add_text(
            f"Frame: {i}", position="upper_right", font_size=20
        )
        update_slice(i)
        if move_camera:
            plotter.camera.azimuth = plotter.camera.azimuth - 1
        plotter.write_frame()
    plotter.close()
