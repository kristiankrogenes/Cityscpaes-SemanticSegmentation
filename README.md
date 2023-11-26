
SSH CONNECTION FOR ALLOCATED NODE:
$ `ssh -L 8888:127.0.0.1:88888 -J kristiwk@idun-login1.hpc.ntnu.no kristiwk@idun-xx-xx`

MODULES FOR IDUN CLUSTER:
# module load intel/2022.05
# module load Python/3.10.4-GCCcore-11.3.0

labelID     classID     categoryID      class           category
0           7           0               road            flat
1           8           0               sidewalk        flat
2           11          3               building        construction
3           12          3               wall            construction
4           13          3               fence           construction      
5           17          4               pole            object
6           19          4               traffic_light   object
7           20          4               traffic_sign    object
8           21          5               vegetation      nature
9           22          5               terrain         nature
10          23          6               sky             sky
11          24          1               person          human
12          25          1               rider           human
13          26          2               car             vehicle
14          27          2               truck           vehicle
15          28          2               bus             vehicle
16          31          2               train           vehicle
17          32          2               motorcycle      vehicle
18          33          2               bicycle         vehicle


ResNet50_bs15_256x512__e10 - Validation
Validation Loss: 2.409107282757759
Validation Accuracy: 0.6328282207250595
Validation IoU Class: 0.46327707823365927
Validation IoU Category: 0.5272995177656412

ResNet50_512x1024__e10 - Validation
Loss:           2.5194937233183912
Accuracy:       0.5421834919273483
IoU_class:      0.3649210102394033
IoU_category:   0.4143423258975433

ResNet101_512x1024__e10 - Validation
Loss:           2.490462741851806
Accuracy:       0.551964100420475
IoU_class:      0.387586262583732
IoU_category:   0.437651407122612

UNet001_bs16_512x1024__e10 - Validation
Loss:           0.8267728711664677
Accuracy:       0.7730995261669159
IoU_class:      0.6400799768567085
IoU_category:   0.7217828414440155

UNet001_bs5_512x1024__e10 - Validation
Loss:           0.6121109865605832
Accuracy:       0.8271327385902405
IoU_class:      0.7132758902907371
IoU_category:   0.7863484336733818

UNet001_bs5_1024x2048__e10- Validation
Loss:           0.1269593373378738
Accuracy:       0.8465072780251502
IoU_class:      0.7399411392807961
IoU_category:   0.8011392827630043

UNet001_bs5_1024x2048__e20- Validation
Loss:           0.3756184733211994
Accuracy:       0.8928921171426774
IoU_class:      0.8115437830090523
IoU_category:   0.8711990822553635

UNet001_bs5_fl_g2_512x1024__e10 - Validation
Loss:           0.443446657054126
Accuracy:       0.689868881255388
IoU_class:      0.535149978324771
IoU_category:   0.641421708762645

UNet001_bs5_fl2_1024x2048__e20 - Validation
Loss:           0.1357104637445882
Accuracy:       0.8443885616660118
IoU_class:      0.7372743279337883
IoU_category:   0.8146708688735962

UNet001_bs5_fl2_1024x2048__e16 - Validation
Loss:           0.1238079238492064
Accuracy:       0.8516810760498047
IoU_class:      0.7480630350112915
IoU_category:   0.8113803888559341