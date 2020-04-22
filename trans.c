template <typename T>
void convert_to_hwc(
    T* chw_data, T* hwc_data, int num, int channel, int height, int width) {
  int chw = channel * height * width;
  int wc = width * channel;
  int index = 0;
  for (int n = 0; n < num; n++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          hwc_data[n * chw + h * wc + w * channel + c] = chw_data[index];
          index++;
        }
      }
    }
  }
}

template <typename T>
void hwc_to_chw(
    T* chw_data, T* hwc_data, int num, int channel, int height, int width) {
  int chw = channel * height * width;
  int wc = width * channel;
  int wh = width * height;
  int index = 0;
  for (int n = 0; n < num; n++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int c = 0; c < channel; c++) {
          chw_data[n * chw + c * wh + h * width + w] = hwc_data[index];
          index++;
        }
      }
    }
  }
}
