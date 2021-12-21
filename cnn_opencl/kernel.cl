// convolution 1 : image data를 3개의 channel로 나눔
__kernel void convolution_1(__global float *inputs,
                            __global float *conv_temp,
                            int outputDim, int N, int input_offset) {
    int id_0 = get_global_id(0);
    int id_1 = get_global_id(1);

    if(id_0 >= (N * N * outputDim)) return;
    
    __global float *input = inputs + input_offset + (N * N * outputDim) * id_1;
    __global float *temp = conv_temp + (N * N * outputDim * 3 * 3) * id_1;

    int a = id_0 / N;
    int b = id_0 % N;
    
    // Convolution Layer에 n개의 필터가 적용된다면 출력 데이터는 n개의 채널을 갖게 됩니다.
    int channel = a / N;
    int channel_a = a - channel * N;
    
    // LOOP UNROLLING
    #pragma unroll
    for(int i = 0; i < 3; i++) {
        int j = 0;
        int x = b + j - 1;
        int y = channel_a + i - 1;
        
        // 삼항연산자 사용을 통한 최적화
        temp[(((channel * 3 * 3) + (3 * i + j++)) * (N * N)) + (channel_a * N + b)]  = (x >= 0 && x < N && y >= 0 && y < N) ? input[((channel * N) + y) * N + x] : 0;
        x = b + j - 1;
        temp[(((channel * 3 * 3) + (3 * i + j++)) * (N * N)) + (channel_a * N + b)]  = (x >= 0 && x < N && y >= 0 && y < N) ? input[((channel * N) + y) * N + x] : 0;
        x = b + j - 1;
        temp[(((channel * 3 * 3) + (3 * i + j++)) * (N * N)) + (channel_a * N + b)]  = (x >= 0 && x < N && y >= 0 && y < N) ? input[((channel * N) + y) * N + x] : 0;
    }
}

// convolution 2 : 입력 데이터가 여러 채널을 갖을 경우 필터는 각 채널을 순회하며 합성곱을 계산한 후, 채널별 피처 맵을 만든다.
__kernel void convolution_2(
        __global float *conv_temp,
        __global float *networks,
        __global float *outputs,
        int inputDim, int outputDim, int nbyn, int filter_offset
    )
{
    int b_i = get_global_id(2);

    const int ROW_A = outputDim;
    const int COL_A = inputDim * 3 * 3;
    const int ROW_B = inputDim * 3 * 3;
    const int COL_B = nbyn * nbyn;

    __global float *temp = conv_temp + (ROW_B * COL_B) * b_i;
    __global float *filter = networks + filter_offset;
    __global float *biases = networks + filter_offset + (ROW_A * COL_A);
    __global float *output = outputs + (ROW_A * COL_B) * b_i;

    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
    
    int lj = get_local_id(0);
    int li = get_local_id(1);
    int gj = get_group_id(0) * TS + lj;
    int gi = get_group_id(1) * TS + li;

    float sum = 0.0f;
    
    // LOOP UNROLLING
    #pragma unroll
    for (int t = 0; t < COL_A; t += TS) {
        
        const int tj = t + lj;
        const int ti = t + li;
        
        // 삼항연산자 사용을 통한 최적화
        Asub[li][lj] = (gi < ROW_A && tj < COL_A) ? filter[gi * COL_A + tj] : 0;
        Bsub[li][lj] = (ti < ROW_B && gj < COL_B) ? temp[ti * COL_B + gj] : 0;

        barrier(CLK_LOCAL_MEM_FENCE);
        
        // LOOP UNROLLING
        #pragma unroll
        for (int k = 0; k < TS; k++) {
            sum += Asub[li][k] * Bsub[k][lj];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (gi < ROW_A && gj < COL_B) {
        output[gi * COL_B + gj] = ReLU(sum + biases[gi]);
    }
}

__kernel void pooling(__global float *inputs,
                      __global float *outputs,
                      int inputDim, int nbyn) {
    int id_0 = get_global_id(0);
    int id_1 = get_global_id(1);

    if(id_0 >= (inputDim * nbyn * nbyn)) return;

    __global float *input = inputs + (id_1 * nbyn * nbyn * inputDim * 4);
    __global float *output = outputs + (id_1 * nbyn * nbyn * inputDim);

    int x = id_0 / (nbyn * nbyn);
    int y = (id_0 / nbyn) % nbyn;
    int z = id_0 % nbyn;

    // local memory
    float max = 0.0f;
    float temp = 0.0f;
    
    // LOOP UNROLLING
    #pragma unroll
    for(int i = 0; i < 2; i++) {
        int j = 0;
        
        // 삼항연산자 사용을 통한 최적화
        temp = input[(x * nbyn * nbyn * 4) + ((y * 2 + i) * 2 * nbyn + z * 2 + j++)];
        max = (max > temp) ? max : temp;
        temp = input[(x * nbyn * nbyn * 4) + ((y * 2 + i) * 2 * nbyn + z * 2 + j++)];
        max = (max > temp) ? max : temp;
    }
    
    // 한번에 저장
    output[(x * nbyn * nbyn) + (y * nbyn + z)] = max;
}

__kernel void fc(__global float *inputs,
                 __global float *outputs,
                 __global float *networks,
                 int inputDim, int outputDim, int networks_offset) {
    int id_0 = get_global_id(0);
    int id_1 = get_global_id(1);

    if(id_0 >= outputDim) return;

    __global float *input = inputs + (id_1 * inputDim);
    __global float *output = outputs + (id_1 * outputDim);
    __global float *filter = networks + networks_offset;
    __global float * biases = networks + networks_offset + (inputDim * outputDim);

    // local memory
    float sum = 0.0f;
    
    // LOOP UNROLLING
    #pragma unroll
    for (int i = 0; i < inputDim; i++)
        sum += input[i] * filter[id_0 * inputDim + i];

    sum += biases[id_0];
    // 한번에 저장
    output[id_0] = ReLU(sum);
}
