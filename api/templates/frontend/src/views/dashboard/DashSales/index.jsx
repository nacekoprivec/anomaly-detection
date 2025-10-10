import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Spinner, Accordion } from 'react-bootstrap';
import IconButton from '@mui/material/IconButton';
import DeleteIcon from '@mui/icons-material/Delete';
import Button from '@mui/material/Button';
import AddIcon from '@mui/icons-material/Add';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import TextField from '@mui/material/TextField';
import api from '../../../api';

//-----------------------|| DASHBOARD SALES ||-----------------------//
export default function DashSales() {
  const [selectedMethod, setSelectedMethod] = useState('border_check.json');
  const [config, setConfig] = useState(null);
  const [overrides, setOverrides] = useState({});
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [detectors, setDetectors] = useState([]);

  // Fetch detectors on mount
  const fetchDetectors = async () => {
    try {
      const res = await api.get('/detectors');
      setDetectors(res.data);
    } catch {
      setDetectors([]);
    }
  };

  useEffect(() => {
    fetchDetectors();
  }, []);

  // Config dropdown component
  function ConfigDropdown({ selectedMethod, setSelectedMethod }) {
    const [availableConfigs, setAvailableConfigs] = useState([]);

    useEffect(() => {
      async function fetchAvailableConfigs() {
        try {
          const res = await api.get('/available_configs');
          setAvailableConfigs(res.data);
        } catch {
          setAvailableConfigs([]);
        }
      }
      fetchAvailableConfigs();
    }, []);

    return (
      <select
        value={selectedMethod}
        onChange={(e) => setSelectedMethod(e.target.value)}
        className="form-control mb-2"
      >
        {availableConfigs.map((ac) => (
          <option key={ac.filename} value={ac.filename}>
            {ac.filename}
          </option>
        ))}
      </select>
    );
  }

  function getNestedValue(obj, path, fallback) {
    return path.split('.').reduce((acc, k) => (acc ? acc[k] : undefined), obj) ?? fallback;
  }

  function ConfigEditor({ data, overrides, setOverrides, parentKey = '' }) {
    const handleChange = (key, value) => {
      setOverrides((prev) => {
        const keys = parentKey ? parentKey.split('.') : [];
        let updated = { ...prev };
        let ref = updated;
        keys.forEach((k) => {
          ref[k] = { ...ref[k] };
          ref = ref[k];
        });
        ref[key] = value;
        return updated;
      });
    };

    return (
      <div style={{ paddingLeft: parentKey ? 15 : 0, borderLeft: parentKey ? '1px solid #eee' : 'none' }}>
        {Object.entries(data).map(([key, value]) => {
          const path = parentKey ? `${parentKey}.${key}` : key;

          if (typeof value === 'string' || typeof value === 'number') {
            return (
              <div className="mb-2" key={path}>
                <label>{key}</label>
                <input
                  type="text"
                  className="form-control"
                  value={getNestedValue(overrides, path, value)}
                  onChange={(e) => handleChange(key, e.target.value)}
                />
              </div>
            );
          }

          if (Array.isArray(value)) {
            if (value.every((v) => typeof v === 'object' && v !== null)) {
              return (
                <div key={path} className="mb-2">
                  <label>{key}</label>
                  {value.map((item, index) => (
                    <ConfigEditor
                      key={`${path}.${index}`}
                      data={item}
                      overrides={overrides?.[key]?.[index] ?? item}
                      setOverrides={(newOverrides) => {
                        setOverrides((prev) => ({
                          ...prev,
                          [key]: [...(prev?.[key] ?? []).slice(0, index), newOverrides, ...(prev?.[key] ?? []).slice(index + 1)]
                        }));
                      }}
                      parentKey={`${path}.${index}`}
                    />
                  ))}
                </div>
              );
            } else {
              return (
                <div className="mb-2" key={path}>
                  <label>{key}</label>
                  <input
                    type="text"
                    className="form-control"
                    value={overrides?.[key]?.join(',') ?? value.join(',')}
                    onChange={(e) => handleChange(key, e.target.value.split(','))}
                  />
                </div>
              );
            }
          }

          if (typeof value === 'object' && value !== null) {
            return (
              <div key={path} className="mb-2">
                <label>{key}</label>
                <ConfigEditor
                  data={value}
                  overrides={overrides?.[key] ?? value}
                  setOverrides={(newOverrides) => {
                    setOverrides((prev) => ({ ...prev, [key]: newOverrides }));
                  }}
                  parentKey={path}
                />
              </div>
            );
          }

          return null;
        })}
      </div>
    );
  }

  // Detector Card Component
  function DetectorCard({ detector, fetchDetectors }) {
    const [selectedMethod, setSelectedMethod] = useState(detector.config_name);
    const [config, setConfig] = useState(null);
    const [overrides, setOverrides] = useState({});
    const [response, setResponse] = useState('');
    const [loading, setLoading] = useState(false);
    const [timestamp, setTimestamp] = useState('');
    const [ftrVector, setFtrVector] = useState('');

    useEffect(() => {
      async function fetchConfig() {
        try {
          const res = await api.get(`/configuration/${selectedMethod}`);
          setConfig(res.data);
          setOverrides(res.data);
        } catch {
          setConfig(null);
        }
      }
      fetchConfig();
    }, [selectedMethod]);

    const handleSaveConfig = async () => {
      try {
        const res = await api.post(`/configuration/${selectedMethod}`, overrides, detector.id);
        setResponse(res.data);
      } catch (error) {
        setResponse('Error: ' + error.message);
      }
    };

    const handleRun = async () => {
      setLoading(true);
      try {
        const res = await api.post(
          `/detectors/${detector.id}/detect_anomaly/?timestamp=${encodeURIComponent(timestamp)}&ftr_vector=${encodeURIComponent(ftrVector)}`
        );
        setResponse(res.data);
      } catch (error) {
        setResponse('Error: ' + error.message);
      }
      setLoading(false);
    };

    return (
      <Card className="mb-3">
        <Card.Body>
          <Accordion flush>
            <Accordion.Item eventKey="0">
              <Accordion.Header>
                <div className="d-flex justify-content-between align-items-center w-100">
                  <span>
                    <strong>{detector.name}</strong> Detector
                  </span>
                  <span
                    className={`badge ${
                      detector.status === 'active'
                        ? 'bg-success'
                        : detector.status === 'error'
                        ? 'bg-danger'
                        : 'bg-secondary'
                    }`}
                  >
                    {detector.status}
                  </span>
                </div>
              </Accordion.Header>
              <Accordion.Body>
                <ConfigDropdown selectedMethod={selectedMethod} setSelectedMethod={setSelectedMethod} />

                <div className="mb-2">
                  <label>Timestamp</label>
                  <input
                    type="text"
                    className="form-control"
                    placeholder='e.g. "123.456"'
                    value={timestamp}
                    onChange={(e) => setTimestamp(e.target.value)}
                  />
                </div>
                <div className="mb-2">
                  <label>Feature Vector</label>
                  <input
                    type="text"
                    className="form-control"
                    placeholder="e.g. 1,2,3,4"
                    value={ftrVector}
                    onChange={(e) => setFtrVector(e.target.value)}
                  />
                </div>

                {config && <ConfigEditor data={config} overrides={overrides} setOverrides={setOverrides} />}

                <div className="mb-3 d-flex align-items-center">
                  <button className="btn btn-success me-2" onClick={handleSaveConfig}>
                    Save Config
                  </button>

                  <button
                    className={`btn ${detector.status === 'inactive' ? 'btn-success' : 'btn-danger'} me-2`}
                    onClick={async () => {
                      try {
                        const newStatus = detector.status === 'inactive' ? 'active' : 'inactive';
                        await api.put(`/detectors/${detector.id}/${newStatus}`);
                        fetchDetectors();
                      } catch (error) {
                        console.error(error);
                      }
                    }}
                  >
                    {detector.status === 'inactive' ? 'Activate' : 'Deactivate'}
                  </button>

                  <Button
                    startIcon={<DeleteIcon />}
                    color="error"
                    onClick={async () => {
                      try {
                        if (confirm('Are you sure you want to delete this detector?')) {
                          await api.delete(`/detectors/${detector.id}`);
                          fetchDetectors();
                        }
                      } catch (error) {
                        console.error(error);
                      }
                    }}
                    className="me-2"
                  ></Button>

                  {loading ? (
                    <Spinner animation="border" style={{ marginLeft: 'auto' }} />
                  ) : (
                    <button className="btn btn-success ms-auto" onClick={handleRun}>
                      Run
                    </button>
                  )}
                </div>

                {response && (
                  <div className="mt-2">
                    <strong>API Response:</strong> {JSON.stringify(response)}
                  </div>
                )}
              </Accordion.Body>
            </Accordion.Item>
          </Accordion>
        </Card.Body>
      </Card>
    );
  }

  // JSON Popup Button
  function JsonPopupButton({ onSave, fetchDetectors }) {
    const [open, setOpen] = useState(false);
    const [jsonText, setJsonText] = useState('{}');
    const [error, setError] = useState('');

    const handleOpen = () => {
      setOpen(true);
      setJsonText('{"name":"a","config_name":"border_check.json"}');
    };
    const handleClose = () => {
      setOpen(false);
      setError('');
    };

    const handleSave = async () => {
      try {
        const parsed = JSON.parse(jsonText);
        onSave(parsed);
        await api.post('/detectors/', parsed);
        fetchDetectors();
        setOpen(false);
      } catch (e) {
        setError('Invalid JSON');
      }
    };

    return (
      <>
        <IconButton color="primary" onClick={handleOpen}>
          <AddIcon />
        </IconButton>

        <Dialog open={open} onClose={handleClose} fullWidth maxWidth="sm">
          <DialogTitle>Enter JSON</DialogTitle>
          <DialogContent>
            <TextField
              multiline
              rows={10}
              fullWidth
              variant="outlined"
              value={jsonText}
              onChange={(e) => setJsonText(e.target.value)}
              error={!!error}
              helperText={error || 'Enter valid JSON here'}
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={handleClose}>Cancel</Button>
            <Button variant="contained" color="primary" onClick={handleSave}>
              Save
            </Button>
          </DialogActions>
        </Dialog>
      </>
    );
  }

  return (
    <Row>
      <Col md={12} xl={6} className="mb-3">
        <Card className="flat-card">
          <Card.Body>
            <div className="d-flex align-items-center mb-3">
              <h1 className="card-title me-3">Anomaly Detectors</h1>
              <JsonPopupButton onSave={(json) => console.log(json)} fetchDetectors={fetchDetectors} />
              <Button
                startIcon={<DeleteIcon />}
                color="error"
                onClick={async () => {
                  try {
                    if (confirm('Are you sure you want to delete all detectors?')) {
                      await api.delete('/detectors');
                      fetchDetectors();
                    }
                  } catch (error) {
                    console.error(error);
                  }
                }}
                className="ms-2"
              />
            </div>

            {detectors.map((det) => (
              <DetectorCard key={det.id} detector={det} fetchDetectors={fetchDetectors} />
            ))}
          </Card.Body>
        </Card>
      </Col>
    </Row>
  );
}
