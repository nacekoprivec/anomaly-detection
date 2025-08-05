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

It will allow users to upload data file, allow option to detect streaming data from provided csv file/json or use default ones, (check for filetype, format)      # websites or devices,
It will provide default configurations and allow manual configuration  of those variables, and run anomaly detection. (check valid input)
It will also provide console output for algorithm results(confusion matrix)
It will provide visual output of results (graph)


It might allow user authentication and authorization for secure access to the API.
It might have option of automatic hypertunning based on user provided data and algorithm configuration. 
It might have database to store info

Project structure based on https://github.com/zhanymkanov/fastapi-best-practices?tab=readme-ov-file#project-structure

RESOURCES
https://fastapi.tiangolo.com/tutorial
https://www.youtube.com/watch?v=rvFsGRvj9jo

TEMPLATES
https://themewagon.github.io/DashboardKit