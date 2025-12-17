# :rocket: Project Guide :rocket:

Tooth-NVS mainly uses an end-to-end feedforward model to process captured tooth or oral videos through the following steps: (1) Extracting frames from the video to convert it into a sequence of RGB images with temporal relationships; (2) Using the vggt-x 3D geometric base model to extract camera parameters, image position and rotation parameters, and generate point cloud files; (3) Using 3DGS to jointly optimize the generated point cloud together with image camera parameters to achieve the goal of novel view synthesis (NVS). Currently, attempts are also being made to optimize using the Nope-Nerf method that does not rely on camera parameters, and the specific results are pending updates.

## :page_facing_up: 1.TODO List:page_facing_up:

- [ ] 1.Complete the extended results of the Nope-Nerf parameter-free novel view synthesis method
- [ ] 2.Update the project display results, including the synthesized point clouds, rendered test images, and the results of PSNR, SSIM and LPIPS metrics
- [ ] 3.Try to use mediapipe to segment the tooth area (personally, I feel the effect is not very good)

## :tada: 2.Experiment Results :tada:

We processed four sets of videos, each approximately 20 to 30 seconds long, containing data on teeth and oral cavities for novel view synthesis (NVS). Finally, we obtained a series of effect demonstrations and actual quantitative index test results, including synthesized point clouds, rendering test images, as well as PSNR, SSIM, and LPIPS index results.

| Index | PSNR | SSIM | LPIPS |
|-------|------|------|-------|
|tooth_1(oral)|29.08313751220703|0.9421060085296631|0.391206294298172|
|tooth_2(oral)|28.964609146118164|0.9439239501953125|0.370437353849411|
|tooth_3(tooth)|30.43889808654785|0.9234111309051514|0.4019245207309723|
|tooth_4(tooth)|30.171770095825195|0.9251460433006287|0.40851619839668274|
|tooth_5(tooth)|28.754833221435547|0.9476994276046753|0.3846614956855774|

| Index | Point Cloud | New-Novel Synthetic |
|-------|-------------|---------------------|
|tooth_1|![tooth_1](assets/media/tooth_1.gif)|![tooth_1](assets/tooth_1.gif)|
|tooth_2|![tooth_2](assets/media/tooth_2.gif)|![tooth_2](assets/tooth_2.gif)|
|tooth_3|![tooth_3](assets/media/tooth_3.gif)|![tooth_3](assets/tooth_3.gif)|
|tooth_4|![tooth_4](assets/media/tooth_4.gif)|![tooth_4](assets/tooth_4.gif)|
|tooth_5|![tooth_5](assets/media/tooth_5.gif)|![tooth_5](assets/tooth_5.gif)|
