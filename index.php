<?php
// Read JSON data from file
$json_data = file_get_contents('data.json');
$data = json_decode($json_data, true);

// Extract data for visualization
$accuracy = $data['model_results']['accuracy'];
$rmse = $data['model_results']['rmse'];
$mae = $data['model_results']['mae'];
$f1_score = $data['model_results']['f1_score'];
$auc = $data['model_results']['auc'];

$normal_consumers = $data['stats']['normal_consumers'];
$fraud_consumers = $data['stats']['fraud_consumers'];
$total_consumers = $data['stats']['total_consumers'];
$no_fraud_percentage = $data['stats']['no_fraud_percentage'];
$test_set_no_fraud = $data['stats']['test_set_no_fraud'];
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Evaluation and Consumer Statistics</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f4f7fb;
            color: #333;
            padding: 30px;
        }

        h1 {
            font-size: 36px;
            color: #2d3e50;
            text-align: center;
            margin-bottom: 20px;
        }

        h2 {
            font-size: 28px;
            color: #2d3e50;
            margin-top: 30px;
            margin-bottom: 15px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 40px;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        th {
            background-color: #007bff;
            color: #fff;
        }

        tr:nth-child(even) {
            background-color: #f2f2f2;
        }

        .chart-container {
            max-width: 900px;
            margin: 0 auto;
            text-align: center;
            margin-top: 40px;
        }

        footer {
            text-align: center;
            margin-top: 40px;
            color: #9e9e9e;
            font-size: 14px;
        }

        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }

        .metric-card {
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
            text-align: center;
        }

        .metric-card h3 {
            font-size: 24px;
            margin-bottom: 10px;
            color: #5c6f7a;
        }

        .metric-card p {
            font-size: 18px;
            color: #2d3e50;
        }

        @media screen and (max-width: 768px) {
            .metrics {
                grid-template-columns: 1fr 1fr;
            }

            h1 {
                font-size: 30px;
            }

            h2 {
                font-size: 24px;
            }
        }
    </style>
</head>
<body>

    <h1>Model Evaluation and Consumer Statistics</h1>

    <!-- Model Evaluation Table -->
    <h2>Model Evaluation</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td><?= $accuracy ?>%</td>
        </tr>
        <tr>
            <td>RMSE</td>
            <td><?= $rmse ?></td>
        </tr>
        <tr>
            <td>MAE</td>
            <td><?= $mae ?></td>
        </tr>
        <tr>
            <td>F1 Score</td>
            <td><?= $f1_score ?>%</td>
        </tr>
        <tr>
            <td>AUC</td>
            <td><?= $auc ?>%</td>
        </tr>
    </table>

    <!-- Consumer Data Metrics -->
    <h2>Consumer Data</h2>
    <div class="metrics">
        <div class="metric-card">
            <h3>Normal Consumers</h3>
            <p><?= $normal_consumers ?></p>
        </div>
        <div class="metric-card">
            <h3>Fraud Consumers</h3>
            <p><?= $fraud_consumers ?></p>
        </div>
        <div class="metric-card">
            <h3>Total Consumers</h3>
            <p><?= $total_consumers ?></p>
        </div>
        <div class="metric-card">
            <h3>No Fraud Percentage</h3>
            <p><?= $no_fraud_percentage ?>%</p>
        </div>
        <div class="metric-card">
            <h3>Test Set No Fraud Percentage</h3>
            <p><?= $test_set_no_fraud ?>%</p>
        </div>
    </div>

    <!-- Bar Charts for Evaluation Metrics -->
    <div class="chart-container">
        <h2>Model Evaluation Metrics</h2>
        <canvas id="metricsChart"></canvas>
    </div>

    <!-- Footer -->
    <footer>
        <p>&copy; 2024 Model Evaluation and Consumer Statistics</p>
    </footer>

    <script>
        // Bar Chart for Evaluation Metrics
        const ctx = document.getElementById('metricsChart').getContext('2d');
        const metricsChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Accuracy', 'F1 Score', 'AUC'],
                datasets: [{
                    label: 'Model Evaluation Metrics',
                    data: [<?= $accuracy ?>, <?= $f1_score ?>, <?= $auc ?>],
                    backgroundColor: '#007bff',
                    borderColor: '#007bff',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(tooltipItem) {
                                return tooltipItem.label + ': ' + tooltipItem.raw + '%';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    </script>

</body>
</html>
