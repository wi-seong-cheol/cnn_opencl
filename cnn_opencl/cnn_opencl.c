
#include "cnn.h"
#include <stdio.h>
#include <time.h>

#define BATCH   100    // batch size
#define TS      16     // tile size

cl_int error;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
char *kernel_source;
size_t kernel_source_size;
cl_program program;
cl_kernel kernel_convolution1, kernel_convolution2;
cl_kernel kernel_pooling, kernel_fc;

cl_mem buf_images;
cl_mem buf_networks;
cl_mem buf_conv_input, buf_conv_temp, buf_conv_output;
cl_mem buf_pooling_input, buf_pooling_output;
cl_mem buf_fc_input, buf_fc_output;
int input_offset, filter_offset;

size_t global_size[3] = { 0, BATCH , 1};
size_t local_size[3] = { 256, 1, 1 };

void build_error(cl_int err) {
   if (err == CL_BUILD_PROGRAM_FAILURE) {
      size_t log_size;
      char* log;

      err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
      CHECK_ERROR(err);

      log = (char*)malloc(log_size + 1);
      err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
      CHECK_ERROR(err);

      log[log_size] = '\0';
      printf("Compiler error:\n%s\n", log);
      free(log);
      exit(0);
   };
}

char* get_source_code(const char* file_name, size_t* len) {
    FILE* file = fopen(file_name, "r");

    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    size_t length = (size_t)ftell(file);
    rewind(file);

    char* source_code = (char*)malloc(length + 1);
    fread(source_code, length, 1, file);
    source_code[length] = '\0';
    fclose(file);
    *len = length;

    return source_code;
}

void convolution(cl_mem* inputs, cl_mem* outputs, cl_mem* networks, int i) {

    filter_offset += (3 * 3 * INPUT_DIM[i - 1] * OUTPUT_DIM[i - 1]) + OUTPUT_DIM[i - 1];
    global_size[0] = NBYN[i] * NBYN[i] * OUTPUT_DIM[i];
    global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
    // Convolution1
    error = clSetKernelArg(kernel_convolution1, 0, sizeof(cl_mem), inputs); CHECK_ERROR(error);
    error = clSetKernelArg(kernel_convolution1, 1, sizeof(cl_mem), &buf_conv_temp); CHECK_ERROR(error);
    error = clSetKernelArg(kernel_convolution1, 2, sizeof(int), &OUTPUT_DIM[i]); CHECK_ERROR(error);
    error = clSetKernelArg(kernel_convolution1, 3, sizeof(int), &NBYN[i]); CHECK_ERROR(error);
    error = clSetKernelArg(kernel_convolution1, 4, sizeof(int), &input_offset); CHECK_ERROR(error);

    error = clEnqueueNDRangeKernel(queue, kernel_convolution1, 2, NULL, global_size, local_size, 0, NULL, NULL); CHECK_ERROR(error);

    // tiling
    global_size[0] = NBYN[i] * NBYN[i];
    global_size[1] = OUTPUT_DIM[i];
    global_size[2] = BATCH;
    local_size[0] = TS;
    local_size[1] = TS;
    local_size[2] = 1;

    global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
    global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1] * local_size[1];

    // Convolution2
    error = clSetKernelArg(kernel_convolution2, 0, sizeof(cl_mem), &buf_conv_temp); CHECK_ERROR(error);
    error = clSetKernelArg(kernel_convolution2, 1, sizeof(cl_mem), networks); CHECK_ERROR(error);
    error = clSetKernelArg(kernel_convolution2, 2, sizeof(cl_mem), outputs); CHECK_ERROR(error);
    error = clSetKernelArg(kernel_convolution2, 3, sizeof(int), &INPUT_DIM[i]); CHECK_ERROR(error);
    error = clSetKernelArg(kernel_convolution2, 4, sizeof(int), &OUTPUT_DIM[i]); CHECK_ERROR(error);
    error = clSetKernelArg(kernel_convolution2, 5, sizeof(int), &NBYN[i]); CHECK_ERROR(error);
    error = clSetKernelArg(kernel_convolution2, 6, sizeof(int), &filter_offset); CHECK_ERROR(error);

    error = clEnqueueNDRangeKernel(queue, kernel_convolution2, 3, NULL, global_size, local_size, 0, NULL, NULL); CHECK_ERROR(error);

    global_size[1] = BATCH;
    global_size[2] = 1;
    local_size[0] = 256;
    local_size[1] = 1;
}

void max_pooling(cl_mem* inputs, cl_mem* outputs, int i) {
    global_size[0] = INPUT_DIM[i] * NBYN[i] * NBYN[i];
    global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];

    error = clSetKernelArg(kernel_pooling, 0, sizeof(cl_mem), inputs); CHECK_ERROR(error)
    error = clSetKernelArg(kernel_pooling, 1, sizeof(cl_mem), outputs); CHECK_ERROR(error);
    error = clSetKernelArg(kernel_pooling, 2, sizeof(int), &INPUT_DIM[i]); CHECK_ERROR(error);
    error = clSetKernelArg(kernel_pooling, 3, sizeof(int), &NBYN[i]); CHECK_ERROR(error);

    error = clEnqueueNDRangeKernel(queue, kernel_pooling, 2, NULL, global_size, local_size, 0, NULL, NULL); CHECK_ERROR(error);
}

void fc_layer(cl_mem* input_neuron, cl_mem* output_neuron, cl_mem* networks, int i) {
    error = clSetKernelArg(kernel_fc, 0, sizeof(cl_mem), input_neuron); CHECK_ERROR(error);
    error = clSetKernelArg(kernel_fc, 1, sizeof(cl_mem), output_neuron); CHECK_ERROR(error);
    error = clSetKernelArg(kernel_fc, 2, sizeof(cl_mem), networks); CHECK_ERROR(error);
    error = clSetKernelArg(kernel_fc, 3, sizeof(int), &INPUT_DIM[i]); CHECK_ERROR(error);
    error = clSetKernelArg(kernel_fc, 4, sizeof(int), &OUTPUT_DIM[i]); CHECK_ERROR(error);
    error = clSetKernelArg(kernel_fc, 5, sizeof(int), &filter_offset); CHECK_ERROR(error);
    
    filter_offset += (INPUT_DIM[i] * OUTPUT_DIM[i]) + OUTPUT_DIM[i];
    global_size[0] = OUTPUT_DIM[i];
    global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
    
    error = clEnqueueNDRangeKernel(queue, kernel_fc, 2, NULL, global_size, local_size, 0, NULL, NULL); CHECK_ERROR(error);
}

void cnn_init(void) {
    // Platform ID
    error = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(error);

    // Device ID
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(error);
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(error);

    // Create Context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
    CHECK_ERROR(error);

    // Create Command Queue
    queue = clCreateCommandQueue(context, device, 0, &error);
    CHECK_ERROR(error);

    // Create Program Object
    kernel_source = get_source_code("kernel.cl", &kernel_source_size);
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &error);
    CHECK_ERROR(error);

    // Build Program
    char option[100];
    sprintf(option, "-cl-fast-relaxed-math -D ReLU(x)=(((x)>0)?(x):0) -D TS=%d", TS);
    error = clBuildProgram(program, 1, &device, option, NULL, NULL);
    CHECK_ERROR(error);

    // Create Kernel
    kernel_convolution1 = clCreateKernel(program, "convolution_1", &error);
    CHECK_ERROR(error);
    kernel_convolution2 = clCreateKernel(program, "convolution_2", &error);
    CHECK_ERROR(error);
    kernel_pooling = clCreateKernel(program, "pooling", &error);
    CHECK_ERROR(error);
    kernel_fc = clCreateKernel(program, "fc", &error);
    CHECK_ERROR(error);
}

void cnn(float* images, float** network, int* labels, float* confidences, int num_images) {
    time_t start, end;

    // Create Buffer
    buf_images = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 32 * 32 * 3 * num_images, NULL, &error); CHECK_ERROR(error);
    buf_networks = clCreateBuffer(context, CL_MEM_READ_ONLY, 60980520, NULL, &error); CHECK_ERROR(error);

    buf_conv_input = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 64 * BATCH, NULL, &error); CHECK_ERROR(error);
    buf_conv_temp = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 64 * 3 * 3 * BATCH, NULL, &error); CHECK_ERROR(error);
    buf_conv_output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 64 * BATCH, NULL, &error); CHECK_ERROR(error);

    buf_pooling_input = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 64 * BATCH, NULL, &error); CHECK_ERROR(error);
    buf_pooling_output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 16 * 16 * 64 * BATCH, NULL, &error); CHECK_ERROR(error);

    buf_fc_input = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512 * BATCH, NULL, &error); CHECK_ERROR(error);
    buf_fc_output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512 * BATCH, NULL, &error); CHECK_ERROR(error);

    // Write Buffer
    error = clEnqueueWriteBuffer(queue, buf_images, CL_FALSE, 0, sizeof(float) * 32 * 32 * 3 * num_images, images, 0, NULL, NULL); CHECK_ERROR(error);
    error = clEnqueueWriteBuffer(queue, buf_networks, CL_FALSE, 0, 60980520, *network, 0, NULL, NULL); CHECK_ERROR(error);

    float* fc3 = (float*)malloc(10 * BATCH * sizeof(float));

    start = clock();
    // run network
    for (int i = 0; i < num_images / BATCH; ++i) {

        // Convolution Layer & Pooling Layer
        input_offset = i * 3 * 32 * 32 * BATCH;
        filter_offset = 0;
        convolution(&buf_images, &buf_conv_output, &buf_networks, 0);
        input_offset = 0;
        convolution(&buf_conv_output, &buf_conv_input, &buf_networks, 1);
        max_pooling(&buf_conv_input, &buf_pooling_output, 2);

        convolution(&buf_pooling_output, &buf_conv_output, &buf_networks, 3);
        convolution(&buf_conv_output, &buf_conv_input, &buf_networks, 4);
        max_pooling(&buf_conv_input, &buf_pooling_output, 5);

        convolution(&buf_pooling_output, &buf_conv_output, &buf_networks, 6);
        convolution(&buf_conv_output, &buf_conv_input, &buf_networks, 7);
        convolution(&buf_conv_input, &buf_conv_output, &buf_networks, 8);
        max_pooling(&buf_conv_output, &buf_pooling_output, 9);

        convolution(&buf_pooling_output, &buf_conv_output, &buf_networks, 10);
        convolution(&buf_conv_output, &buf_conv_input, &buf_networks, 11);
        convolution(&buf_conv_input, &buf_conv_output, &buf_networks, 12);
        max_pooling(&buf_conv_output, &buf_pooling_output, 13);

        convolution(&buf_pooling_output, &buf_conv_output, &buf_networks, 14);
        convolution(&buf_conv_output, &buf_conv_input, &buf_networks, 15);
        convolution(&buf_conv_input, &buf_conv_output, &buf_networks, 16);
        max_pooling(&buf_conv_output, &buf_pooling_output, 17);

        // FC layer
        filter_offset += (3 * 3 * INPUT_DIM[17] * OUTPUT_DIM[17]) + OUTPUT_DIM[17];
        fc_layer(&buf_pooling_output, &buf_fc_output, &buf_networks, 18);
        fc_layer(&buf_fc_output, &buf_fc_input, &buf_networks, 19);
        fc_layer(&buf_fc_input, &buf_fc_output, &buf_networks, 20);

        error = clEnqueueReadBuffer(queue, buf_fc_output, CL_TRUE, 0, sizeof(float) * 10 * BATCH, fc3, 0, NULL, NULL); CHECK_ERROR(error);

        float* backup = fc3;
        for (int j = 0; j < BATCH; j++) {
            // Softmax
            softmax(fc3, 10);

            // Find max
            labels[i * BATCH + j] = find_max(fc3, 10);
            confidences[i * BATCH + j] = fc3[labels[i * BATCH + j]];
            fc3 += 10;
        }
        fc3 = backup;
    }

    end = clock();
    printf("OpenCl Elapsed time: %.6f sec\n", (double)(end - start) / CLOCKS_PER_SEC);

    clReleaseMemObject(buf_images);
    clReleaseMemObject(buf_networks);
    clReleaseMemObject(buf_conv_input);
    clReleaseMemObject(buf_conv_temp);
    clReleaseMemObject(buf_conv_output);
    clReleaseMemObject(buf_pooling_input);
    clReleaseMemObject(buf_pooling_output);
    clReleaseMemObject(buf_fc_input);
    clReleaseMemObject(buf_fc_output);
    clReleaseKernel(kernel_convolution1);
    clReleaseKernel(kernel_convolution2);
    clReleaseKernel(kernel_pooling);
    clReleaseKernel(kernel_fc);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(fc3);
}
