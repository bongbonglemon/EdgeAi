# Project Write-Up

## How I convert with Model Optimizer

I download the SSD MobileNet V2 COCO model from here:http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

This is the code I ran:
```
/opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```


## Explaining Custom Layers

To convert custom layers, we need to register them as extensions to the Model Optimizer. In the project, it looked like this:

```
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
plugin.add_extension(CPU_EXTENSION, "CPU")
```

We have to have to handle custom layers because the Model Optimizers does not know how to convert them. They may be layers we may have implemented ourselves from research published yesterday.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were to first use load the tensorflow model and make the inference and then take that same model, put it through the Model Optimizer and Inference Engine and get that inference result.

The difference between model accuracy pre- and post-conversion was that the one before conversion did slightly better. The model that went through post-conversion had boxes disappearing in some frames leading to double counting in the total counted.

The size of the model pre- and post-conversion was the post-conversion one was smaller. The model pre-conversion was 69.7Mb but only 112kb post-conversion which was got from looking at the file sizes which I've seen others to have used as good estimates.

The inference time of the model pre- and post-conversion was the post-conversion one was faster. The pre-conversion took about 12 seconds each frame while the post-conversion took about 0.07 seconds. I just used Python's time module. I recorded a start and end time before and after the inference is made and then got the difference.

Using this AI at this Edge in this scenario saves up money in using cloud computing to do inference as well as time that may take in sending up videos to the cloud for inference. At the Edge, we can straight away do the inference and only have to send small data like number of people counted. We also get more privacy because the videos of people is less easily hacked if not on the cloud.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are to limit the number of people in area and survey the popularity of a meuseum exhibit.

Each of these use cases would be useful because limiting the number of people in a small area can help facilliate social distancing in times such as now and knowing the popularity of a meuseum exhibit will inform the meuseum owners on what artifacts/art pieces to display to increase revenue.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows: the end user has to deploy the model in a well lit area and of certain distance from the presence of people. Ideally it should have a clear and full view of the person in order to detect that person.
