
TO RUN
```
docker run --rm track3-cbam

docker run --rm \
  -v $(pwd)/docker_output:/app/test_images_heatmap \
  -v $(pwd)/docker_output_cbam:/app/test_images_heatmap_cbam \
  -v $(pwd)/docker_output_yolo:/app/test_images_yolo \
  -v $(pwd)/docker_output_cbam_det:/app/test_images_cbam \
  track3-cbam
  ```