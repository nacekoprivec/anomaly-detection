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

import { Spinner } from "react-bootstrap";


// -----------------------|| DASHBOARD SALES ||-----------------------//
export default function DashSales() {
  const [selectedMethod, setSelectedMethod] = useState('border_check.json');
  const [config, setConfig] = useState(null);
  const [overrides, setOverrides] = useState({});
  const [response, setResponse] = useState('');
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

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

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
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

  const handleUpload = async () => {
    if (!file) {
      setResponse("Please select a file first.");
      return;
    }

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("config_name", selectedMethod);

      const res = await api.post("/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setResponse(`Upload successful: ${JSON.stringify(res.data)}`);
    } catch (error) {
      setResponse("Error: " + error.message);
    }
  };




  return (
    <Row>
      <Col md={12} xl={6} className="mb-3">
        <Card className="flat-card">
          <div className="row-table">
            <Card.Body className="col-sm-6 br">
              <h1 className="mb-3">Anomaly Detection Dashboard</h1>
              <div className="mb-3">
                <select
                  value={selectedMethod}
                  onChange={handleSelectChange}
                  className="form-control mb-2"
                >
                  <option value="border_check.json">BorderCheck</option>
                  <option value="clustering.json">Clustering</option>
                  <option value="cumulative.json">Cumulative</option>
                  <option value="ema_percentile.json">EMAPercentile</option>
                  <option value="ema.json">EMA</option>
                  <option value="filtering.json">Filtering</option>
                  <option value="gan.json">GAN</option>
                  <option value="hampel.json">Hampel</option>
                  <option value="isolation_forest.json">IsolationForest</option>
                  <option value="linear_fit.json">LinearFit</option>
                  <option value="macd.json">MACD</option>
                  <option value="pca.json">PCA</option>
                  <option value="rrcf_trees.json">RRCF</option>
                  <option value="trend_classification.json">TrendClassification</option>
                  <option value="welford.json">Welford</option>
                </select>
                {config && (
                  <div>
                    {/* Render all top-level config fields */}
                    {Object.entries(config).map(([key, value]) => {
                      // Handle primitive values
                      if (
                        typeof value === 'string' ||
                        typeof value === 'number'
                      ) {
                        return (
                          <div className="mb-2" key={key}>
                            <label>{key}</label>
                            <input
                              type="text"
                              className="form-control"
                              value={overrides[key] ?? ''}
                              onChange={e => handleConfigChange(key, e.target.value)}
                            />
                          </div>
                        );
                      }
                      // Handle arrays of primitives
                      if (
                        Array.isArray(value) &&
                        value.every(
                          v =>
                            typeof v === 'string' ||
                            typeof v === 'number'
                        )
                      ) {
                        return (
                          <div className="mb-2" key={key}>
                            <label>{key}</label>
                            <input
                              type="text"
                              className="form-control"
                              value={overrides[key]?.join(',') ?? ''}
                              onChange={e =>
                                handleConfigChange(
                                  key,
                                  e.target.value.split(',')
                                )
                              }
                            />
                          </div>
                        );
                      }
                      // Handle arrays of objects
                      if (
                        Array.isArray(value) &&
                        value.every(v => typeof v === 'object')
                      ) {
                        return (
                          <div className="mb-2" key={key}>
                            <label>{key}</label>
                            {value.map((obj, idx) => (
                              <div key={idx} style={{ paddingLeft: 10, borderLeft: '2px solid #eee', marginBottom: 8 }}>
                                {Object.entries(obj).map(([subKey, subValue]) => (
                                  <div key={subKey}>
                                    <label>{subKey}</label> 
                                    <input
                                      type="text"
                                      className="form-control"
                                      value={overrides[key]?.[idx]?.[subKey] ?? subValue ?? ''}
                                      onChange={e => {
                                        // Update nested array of objects
                                        setOverrides(prev => {
                                          const arr = prev[key] ? [...prev[key]] : [...value];
                                          arr[idx] = { ...arr[idx], [subKey]: e.target.value };
                                          return { ...prev, [key]: arr };
                                        });
                                      }}
                                    />
                                  </div>
                                ))}
                              </div>
                            ))}
                          </div>
                        );
                      }
                      // Handle objects
                      if (typeof value === 'object') {
                        return (
                          <div className="mb-2" key={key}>
                            <label>{key}</label>
                            {Object.entries(value).map(([subKey, subValue]) => (
                              <div key={subKey} style={{ paddingLeft: 10 }}>
                                <label>{subKey}</label>
                                <input
                                  type="text"
                                  className="form-control"
                                  value={overrides[key]?.[subKey] ?? subValue ?? ''}
                                  onChange={e => {
                                    setOverrides(prev => ({
                                      ...prev,
                                      [key]: { ...(prev[key] || value), [subKey]: e.target.value }
                                    }));
                                  }}
                                />
                              </div>
                            ))}
                          </div>
                        );
                      }
                      return null;
                    })}
                  </div>
                )}
                
                <div className="mb-3 d-flex align-items-center">
                  <button className="btn btn-success" onClick={handleSaveConfig}>
                  Save Config
                  </button>
                
                <div>
                  <button className="btn btn-primary ms-2" onClick={handleUpload}>
                    Upload
                  </button>
                  <input type="file" accept=".csv" onChange={handleFileChange} />
                </div>

                {loading ? (
              <Spinner animation="border" style={{ marginLeft: 'auto' }} />) : (
              <button
                  className="btn btn-success ms-auto"
                  onClick={handleRun}
                  style={{ marginLeft: 'auto' }}
                >
                  Run
                </button> )}

                </div>
                {response && (
                  <div className="mt-2">
                    <strong>API Response:</strong> {JSON.stringify(response)}
                  </div>
                )}
              </div>
            </Card.Body>
          </div>
        </Card>
      </Col>
      <Col md={12} xl={6}>
        <Card className="flat-card">
          <div className="row-table">
            <Card.Body className="col-sm-6 br">
              <FlatCard params={{ title: 'Customers', iconClass: 'text-primary mb-1', icon: 'group', value: '1000' }} />
            </Card.Body>
            <Card.Body className="col-sm-6 d-none d-md-table-cell d-lg-table-cell d-xl-table-cell card-body br">
              <FlatCard params={{ title: 'Revenue', iconClass: 'text-primary mb-1', icon: 'language', value: '1252' }} />
            </Card.Body>
            <Card.Body className="col-sm-6 card-bod">
              <FlatCard params={{ title: 'Growth', iconClass: 'text-primary mb-1', icon: 'unarchive', value: '600' }} />
            </Card.Body>
          </div>
          <div className="row-table">
            <Card.Body className="col-sm-6 br">
              <FlatCard
                params={{
                  title: 'Returns',
                  iconClass: 'text-primary mb-1',
                  icon: 'swap_horizontal_circle',
                  value: '3550'
                }}
              />
            </Card.Body>
            <Card.Body className="col-sm-6 d-none d-md-table-cell d-lg-table-cell d-xl-table-cell card-body br">
              <FlatCard params={{ title: 'Downloads', iconClass: 'text-primary mb-1', icon: 'cloud_download', value: '3550' }} />
            </Card.Body>
            <Card.Body className="col-sm-6 card-bod">
              <FlatCard params={{ title: 'Order', iconClass: 'text-primary mb-1', icon: 'shopping_cart', value: '100%' }} />
            </Card.Body>
          </div>
        </Card>
        <Row>
          <Col md={6}>
            <Card className="support-bar overflow-hidden">
              <Card.Body className="pb-0">
                <h2 className="m-0">53.94%</h2>
                <span className="text-primary">Conversion Rate</span>
                <p className="mb-3 mt-3">Number of conversions divided by the total visitors. </p>
              </Card.Body>
              <Chart {...SalesSupportChartData()} />
              <Card.Footer className="border-0 bg-primary text-white background-pattern-white">
                <Row className="text-center">
                  <Col>
                    <h4 className="m-0 text-white">10</h4>
                    <span>2018</span>
                  </Col>
                  <Col>
                    <h4 className="m-0 text-white">15</h4>
                    <span>2017</span>
                  </Col>
                  <Col>
                    <h4 className="m-0 text-white">13</h4>
                    <span>2016</span>
                  </Col>
                </Row>
              </Card.Footer>
            </Card>
          </Col>
          <Col md={6}>
            <Card className="support-bar overflow-hidden">
              <Card.Body className="pb-0">
                <h2 className="m-0">1432</h2>
                <span className="text-primary">Order Delivered</span>
                <p className="mb-3 mt-3">Number of conversions divided by the total visitors. </p>
              </Card.Body>
              <Card.Footer className="border-0">
                <Row className="text-center">
                  <Col>
                    <h4 className="m-0">130</h4>
                    <span>May</span>
                  </Col>
                  <Col>
                    <h4 className="m-0">251</h4>
                    <span>June</span>
                  </Col>
                  <Col>
                    <h4 className="m-0 ">235</h4>
                    <span>July</span>
                  </Col>
                </Row>
              </Card.Footer>
              <Chart type="bar" {...SalesSupportChartData1()} />
            </Card>
          </Col>
        </Row>
      </Col>
      <Col md={12} xl={6}>
        <Card>
          <Card.Header>
            <h5>Department wise monthly sales report</h5>
          </Card.Header>
          <Card.Body>
            <Row className="pb-2">
              <div className="col-auto m-b-10">
                <h3 className="mb-1">$21,356.46</h3>
                <span>Total Sales</span>
              </div>
              <div className="col-auto m-b-10">
                <h3 className="mb-1">$1935.6</h3>
                <span>Average</span>
              </div>
            </Row>
            <Chart {...SalesAccountChartData()} />
          </Card.Body>
        </Card>
      </Col>
      <Col md={12} xl={6}>
        <Card>
          <Card.Body>
            <h6>Customer Satisfaction</h6>
            <span>It takes continuous effort to maintain high customer satisfaction levels Internal and external.</span>
            <Row className="d-flex justify-content-center align-items-center">
              <Col>
                <Chart type="pie" {...SalesCustomerSatisfactionChartData()} />
              </Col>
            </Row>
          </Card.Body>
        </Card>
        {/* Product Table */}
        <ProductTable {...productData} />
      </Col>
      <Col md={12} xl={6}>
        <Row>
          <Col sm={6}>
            <ProductCard params={{ title: 'Total Profit', primaryText: '$1,783', icon: 'card_giftcard' }} />
          </Col>
          <Col sm={6}>
            <ProductCard params={{ variant: 'primary', title: 'Total Orders', primaryText: '15,830', icon: 'local_mall' }} />
          </Col>
          <Col sm={6}>
            <ProductCard params={{ variant: 'primary', title: 'Average Price', primaryText: '$6,780', icon: 'monetization_on' }} />
          </Col>
          <Col sm={6}>
            <ProductCard params={{ title: 'Product Sold', primaryText: '6,784', icon: 'local_offer' }} />
          </Col>
        </Row>
        {/* Feed Table */}
        <FeedTable {...feedData} />
      </Col>
    </Row>
  );
}
