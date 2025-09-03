TODO: 
Fix gan, trend_C, 
Name of detectors should be unique
add status
change log config when modifying detector

After 10s go through all anomaly detectors and triger test class.
Fetch data from website (ID, create your own date, vodostaj). 
Process datapoint.
Send proccesed datapoint to anomaly detector, which updates everything for its log and adds datapoint to datapoint class.
Pause 10s, loop until stop key is pressed


Tasks:
- Fetch streaming data from http://hmljn.arso.gov.si/vode/podatki/stanje_voda_samodejne.html (UNIX time)
    - Exctract timestamp, vodostaj
- Modify existing anomaly detection to read and process streaming data
- Modify API so it works with streaming data
    - Modify database so log it stores TP, FP, TN, FN.
    - Add isAnomaly to each datapoint
    - Rename anomaly to DataPoint (also change relationship)
    - Each detector should have ID, status, created_timedate. Store in database (also make CRUD)
    - Execute migrations
- Make multiple running instances of anomaly detectors avaliable at the same time 
- Dashboard should have anomaly detectors list, which enables you to see each instance
- Logs tab stores past runs


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
It will have database to store info such as table anomaly(id, timestamp,ftr_vector), inside log table confusion matrix, start time, end time, configuration used as string. You will be able to see logs and all anomalies for each log. Optionally you will be able to read and delete logs.


It might allow user authentication and authorization for secure access to the API.
It might have option of automatic hypertunning based on user provided data and algorithm configuration. 

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