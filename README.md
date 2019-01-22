# docker-ml
A Machine Learning service in a Docker

## Run service
Start service with the command below and it running on http://localhost:7000/

`$ docker-compose up`

## Train Model
Run the following command to train the classification model.

`$ python api/ml/main.py --resume --lr=0.01`

- resume:resume from checkpoint
- lr:learning rate

## Classification
Make a HTTP Post with the picture you want to classification using below url, the service will return the classification result.

### URL

`[POST]  http://localhost:7000/predict`

### Request Header

Field | Description |
----- | ----------- |
Content-Type | multipart/form-data |

### Request Body

Field | Description |
----- | ----------- |
file | The picture you want to classification |

### Example

```
POST /predict HTTP/1.1
Host: localhost:7000
Cache-Control: no-cache
Postman-Token: b4438d09-ee3e-aa3d-96e2-279042e53c02
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW

------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="file"; filename="dog.jpeg"
Content-Type: image/jpeg


------WebKitFormBoundary7MA4YWxkTrZu0gW--
```
