document.getElementById('predictForm').addEventListener('submit', function(event) {
    event.preventDefault();  // Prevent the form from reloading the page

    // Get values from the input fields
    const feature1 = parseFloat(document.getElementById('feature1').value);
    const feature2 = parseFloat(document.getElementById('feature2').value);

    // Prepare the data to send to the backend API
    const data = {
        feature1: feature1,
        feature2: feature2
    };

    // Make the POST request to the backend API
    fetch('https://ml-api-416217366305.europe-southwest1.run.app/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        // Display the prediction result
        const prediction = data.prediction === 1 ? "Positive (1)" : "Negative (0)";
        document.getElementById('predictionResult').innerText = `Prediction: ${prediction}`;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('predictionResult').innerText = "Error: Could not fetch prediction.";
    });
});
