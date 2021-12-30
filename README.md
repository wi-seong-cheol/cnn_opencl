# cnn_opencl
#### CIFAR-10 데이터셋을 사용한 VGG16 Model을 OpenCL로 구현한 것입니다.

#### Environment
- 2.3 GHz 8코어 Intel Core i9
- AMD Radeon Pro 5500M
- XCode 13.1 Realse Mode
- DataSet: CIFAR-10

***
#### 적용 기법
- Tiling
- Batch
- Loop Unrolling
- Local Memory 사용
- Covolution layer 병렬처리
- Pooling layer 병렬처리
- FC layer 병렬처리
***
<img src="https://user-images.githubusercontent.com/53855302/147726255-76bf03ba-6977-4ef8-a48d-c6328340f079.jpg">
