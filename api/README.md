TODO: 
strend_C, 
Name of detectors should be unique
add status
change log config when modifying detector
fix confusion matrix


 
Tasks:
- create detector/{id} & modifiable configuration 
- delete detector/{id}
- change detector configuration
- stop
- post(detector/{id}/timestamp&ftr_vector)
    - return is_anomaly and configuration used



#FastAPI guidelines#
Use response model
Use annotated 
Use pydantic
Use inheritance for hiding sensitive info
Use raise, you can have custom ones also instead of internal error
    status_code=status.HTTP_201_CREATED
jsonable_encoder
use Dependency Injection
use decorator dependenceis, they wont be sent to the functionm
yield
await asyncio.sleep(10) # non-blocking I/O operation

run app
conda activate anomaly
uvicorn api.src.main:app --reload

API for Anomaly Detection Algorithms


Project structure based on https://github.com/zhanymkanov/fastapi-best-practices?tab=readme-ov-file#project-structure

RESOURCES
https://fastapi.tiangolo.com/tutorial
https://www.youtube.com/watch?v=rvFsGRvj9jo
https://www.youtube.com/watch?v=E8lXC2mR6-k

https://www.youtube.com/watch?v=aSdVU9-SxH4

TEMPLATES
https://themewagon.github.io/DashboardKit

echo api
https://www.youtube.com/watch?v=-oCHXAUwZt0