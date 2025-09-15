// react-bootstrap
import { Row, Col, Card } from 'react-bootstrap';

// third party
import Chart from 'react-apexcharts';

// project imports
import FlatCard from 'components/Widgets/Statistic/FlatCard';
import ProductCard from 'components/Widgets/Statistic/ProductCard';
import FeedTable from 'components/Widgets/FeedTable';
import ProductTable from 'components/Widgets/ProductTable';
import { SalesCustomerSatisfactionChartData } from './chart/sales-customer-satisfication-chart';
import { SalesAccountChartData } from './chart/sales-account-chart';
import { SalesSupportChartData } from './chart/sales-support-chart';
import { SalesSupportChartData1 } from './chart/sales-support-chart1';
import feedData from 'data/feedData';
import productData from 'data/productTableData';

import React, { useState } from 'react';
import api from '../../../api';
import { useEffect } from 'react';

import { Spinner, Accordion } from "react-bootstrap";

import IconButton from "@mui/material/IconButton";
import DeleteIcon from "@mui/icons-material/Delete";
import Button from '@mui/material/Button';


//-----------------------|| DASHBOARD SALES ||-----------------------//
export default function DashSales() {
  const [selectedMethod, setSelectedMethod] = useState('border_check.json');
  const [config, setConfig] = useState(null);
  const [overrides, setOverrides] = useState({});
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [detectors, setDetectors] = useState([]);

  // Fetch config when method changes
  useEffect(() => {
    async function fetchConfig() {
      try {
        const res = await api.get(`/configuration/${selectedMethod}`);
        setConfig(res.data);
        setOverrides(res.data);
      } catch (error) {
        setConfig(null);
      }
    }
    fetchConfig();
  }, [selectedMethod]);

  // Handle dropdown change
  const handleSelectChange = (e) => setSelectedMethod(e.target.value);

  // Handle config field change
  const handleConfigChange = (key, value) => {
    setOverrides(prev => ({ ...prev, [key]: value }));
  };

  // Send modified config
  const handleSaveConfig = async () => {
    try {
      const res = await api.post(`/configuration/${selectedMethod}`, overrides);
      setResponse(res.data);
    } catch (error) {
      setResponse('Error: ' + error.message);
    }
  };

  const handleRun = async () => {
    setLoading(true);
    try {
      const res = await api.post(`/run/${selectedMethod}`);
      setResponse(res.data);
    } catch (error) {
      setResponse('Error: ' + error.message);
    }
    setLoading(false);
  };

  async function fetchDetectors() {
    try {
      const res = await api.get("/detectors");
      setDetectors(res.data);
    } catch (error) {
      setDetectors([]);
    }
  }

  // run once on mount
  useEffect(() => {
    fetchDetectors();
  }, []);


  function ConfigDropdown({ selectedMethod, setSelectedMethod }) {
    const [availableConfigs, setAvailableConfigs] = useState([]);

    useEffect(() => {
      async function fetchAvailableConfigs() {
        try {
          const res = await api.get("/available_configs");
          setAvailableConfigs(res.data);
        } catch (error) {
          setAvailableConfigs([]);
        }
      }
      fetchAvailableConfigs();
    }, []);

    const handleSelectChange = (e) => {
      setSelectedMethod(e.target.value);
    };

    return (
      <select
        value={selectedMethod}
        onChange={handleSelectChange}
        className="form-control mb-2"
      >
        {availableConfigs.map((ac) => (
          <option key={ac.value} value={ac.value}>
            {ac.name}
          </option>
        ))}
      </select>
    );
  }
  function ConfigEditor({ data, overrides, setOverrides, parentKey = "" }) {
    const handleChange = (key, value) => {
      setOverrides(prev => {
        const keys = parentKey ? parentKey.split(".") : [];
        let updated = { ...prev };

        let ref = updated;
        for (let i = 0; i < keys.length; i++) {
          const k = keys[i];
          ref[k] = { ...ref[k] };
          ref = ref[k];
        }
        ref[key] = value;
        return updated;
      });
    };

    return (
      <div style={{ paddingLeft: parentKey ? 15 : 0, borderLeft: parentKey ? "1px solid #eee" : "none" }}>
        {Object.entries(data).map(([key, value]) => {
          const path = parentKey ? `${parentKey}.${key}` : key;

          if (typeof value === "string" || typeof value === "number") {
            return (
              <div className="mb-2" key={path}>
                <label>{key}</label>
                <input
                  type="text"
                  className="form-control"
                  value={overrides?.[key] ?? value}
                  onChange={e => handleChange(key, e.target.value)}
                />
              </div>
            );
          }

          if (Array.isArray(value)) {
            const isObjectArray = value.every(v => typeof v === "object" && v !== null);

            if (isObjectArray) {
              return (
                <div key={path} className="mb-2">
                  <label>{key}</label>
                  {value.map((item, index) => (
                    <ConfigEditor
                      key={`${path}.${index}`}
                      data={item}
                      overrides={overrides?.[key]?.[index] ?? item}
                      setOverrides={(newOverrides) => {
                        setOverrides(prev => ({
                          ...prev,
                          [key]: [
                            ...(prev?.[key] ?? []).slice(0, index),
                            newOverrides,
                            ...(prev?.[key] ?? []).slice(index + 1)
                          ]
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
                    value={overrides?.[key]?.join(",") ?? value.join(",")}
                    onChange={e => handleChange(key, e.target.value.split(","))}
                  />
                </div>
              );
            }
          }

          if (typeof value === "object" && value !== null) {
            return (
              <div key={path} className="mb-2">
                <label>{key}</label>
                <ConfigEditor
                  data={value}
                  overrides={overrides?.[key] ?? value}
                  setOverrides={(newOverrides) => {
                    setOverrides(prev => ({
                      ...prev,
                      [key]: newOverrides
                    }));
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



  function DetectorCard({ detector }) {
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

    const handleConfigChange = (key, value) => {
      setOverrides(prev => ({ ...prev, [key]: value }));
    };

    const handleSaveConfig = async () => {
      try {
        const res = await api.post(`/configuration/${selectedMethod}`, overrides);
        setResponse(res.data);
      } catch (error) {
        setResponse('Error: ' + error.message);
      }
    };

    const handleRun = async () => {
      setLoading(true);
      try {
        const res = await api.post(`/detectors/${detector.id}/${timestamp}&${ftrVector}`, {
        });
        setResponse(res.data);
      } catch (error) {
        setResponse('Error: ' + error.message);
      }
      setLoading(false);
    };

    return (
      <Card className="mb-3">
        <Card.Body>
          <Accordion alwaysOpen>
            <Accordion.Header>
              <div className="d-flex justify-content-between align-items-center w-100">
                <span>
                  <strong>{detector.name}</strong> Detector
                </span>
                <span
                  className={`badge ${detector.status === "active"
                      ? "bg-success"
                      : detector.status === "error"
                        ? "bg-danger"
                        : "bg-secondary"
                    }`}
                >
                  {detector.status}
                </span>

              </div>
            </Accordion.Header>
            <Accordion.Body> 
              {/* Display detector info */}
              {Object.entries(detector).map(([key, value]) => (
                key !== 'name' && key !== 'status' && key !== 'config' && (
                  <div className="mb-2" key={key}>
                    <strong>{key}:</strong> {JSON.stringify(value)}
                  </div>
                )
              ))}

              {/* Config dropdown */}
              <ConfigDropdown
                selectedMethod={selectedMethod}
                setSelectedMethod={setSelectedMethod}
              />

              {/* Timestamp + feature vector inputs */}
              <div className="mb-2">
                <label>Timestamp</label>
                <input
                  type="text"
                  className="form-control"
                  placeholder='e.g. "123.456"'
                  value={timestamp}
                  onChange={e => setTimestamp(e.target.value)}
                />
              </div>
              <div className="mb-2">
                <label>Feature Vector</label>
                <input
                  type="text"
                  className="form-control"
                  placeholder="e.g. 1,2,3,4"
                  value={ftrVector}
                  onChange={e => setFtrVector(e.target.value)}
                />
              </div>

              {/* Render config editor */}
              {config && (
                <ConfigEditor
                  data={config}
                  overrides={overrides}
                  setOverrides={setOverrides}
                />
              )}

              {/* Actions */}
              <div className="mb-3 d-flex align-items-center">
                <button className="btn btn-success" onClick={handleSaveConfig}>
                  Save Config
                </button>

                <button
                  className={`ms-2 btn ${detector.status === "inactive" ? "btn-success" : "btn-danger"}`}
                  onClick={async () => {
                    try {
                      const newStatus = detector.status === "inactive" ? "active" : "inactive";
                      await api.put(`/detectors/${detector.id}/${newStatus}`);
                      setDetectors(prev =>
                        prev.map(d =>
                          d.id === detector.id ? { ...d, status: newStatus } : d
                        )
                      );
                    } catch (error) {
                      console.error("Error toggling detector status:", error);
                    }
                  }}
                >
                  {detector.status === "inactive" ? "Activate" : "Deactivate"}
                </button>

                <Button startIcon={<DeleteIcon />} color="error" onClick={async () => {
                  try {
                    if (confirm("Are you sure you want to delete this detector?")) {
                      await api.delete(`/detectors/${detector.id}`);
                    }
                  } catch (error) {
                    console.error("Error deleting detector:", error);
                  }
                }}>
                </Button>
                {loading ? (
                  <Spinner animation="border" style={{ marginLeft: 'auto' }} />
                ) : (
                  <button
                    className="btn btn-success ms-auto"
                    onClick={handleRun}
                    style={{ marginLeft: 'auto' }}
                  >
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
          </Accordion>
        </Card.Body>
      </Card>
    );
  }


  return (
    <Row>
      <Col md={12} xl={6} className="mb-3">
        <Card className="flat-card">
          <div className="row-table">
            <Card.Body className="col-sm-12 br">
              {detectors.map(det => (
                <DetectorCard key={det.id} detector={det} />
              ))}
            </Card.Body>
          </div>
        </Card>
      </Col>
    </Row>
  );
}
