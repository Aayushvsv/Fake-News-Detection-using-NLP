document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded');
    
    const newsText = document.getElementById('newsText');
    const classifyBtn = document.getElementById('classifyBtn');
    const resultDiv = document.getElementById('result');
    const resultContent = document.getElementById('resultContent');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');

    console.log('Elements:', { newsText, classifyBtn, resultDiv, loadingDiv, errorDiv });

    classifyBtn.addEventListener('click', classifyNews);

    async function classifyNews() {
        console.log('Classify button clicked');
        const text = newsText.value.trim();
        
        if (!text) {
            console.log('No text entered');
            showError('Please enter some text to classify');
            return;
        }

        // Show loading, hide previous results and errors
        console.log('Showing loading state');
        loadingDiv.classList.remove('hidden');
        resultDiv.classList.add('hidden');
        errorDiv.classList.add('hidden');
        classifyBtn.disabled = true;

        try {
            console.log('Sending request to backend...');
            const response = await fetch('http://localhost:8000/predict/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            console.log('Response status:', response.status);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                console.error('API Error:', errorData);
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Received data:', data);
            displayResult(data);
        } catch (error) {
            console.error('Error in classifyNews:', error);
            showError(error.message || 'An error occurred while classifying the text');
        } finally {
            console.log('Hiding loading state');
            loadingDiv.classList.add('hidden');
            classifyBtn.disabled = false;
        }
    }

    function displayResult(data) {
        console.log('Displaying result:', data);
        
        // Handle different response formats
        let prediction, confidence;
        
        try {
            if (data.prediction !== undefined && data.confidence !== undefined) {
                prediction = data.prediction;
                confidence = data.confidence;
            } else if (Array.isArray(data) && data[0] && data[0].prediction !== undefined) {
                prediction = data[0].prediction;
                confidence = data[0].confidence;
            } else {
                throw new Error('Unexpected response format from server');
            }
    
            const isReal = String(prediction) === '1';
            const confidencePercent = (parseFloat(confidence) * 100).toFixed(2);
            
            console.log(`Prediction: ${isReal ? 'Real' : 'Fake'}, Confidence: ${confidencePercent}%`);
            
            // Update the DOM
            resultDiv.className = isReal ? 'real' : 'fake';
            resultContent.innerHTML = `
                <h3>${isReal ? '✅ Real News' : '❌ Fake News'}</h3>
                <div class="confidence">Confidence: ${confidencePercent}%</div>
            `;
            
            resultDiv.classList.remove('hidden');
            console.log('Result displayed');
        } catch (error) {
            console.error('Error displaying result:', error, 'Data:', data);
            showError('Error displaying the classification result');
        }
    }

    function showError(message) {
        console.error('Showing error:', message);
        errorDiv.textContent = message;
        errorDiv.classList.remove('hidden');
        resultDiv.classList.add('hidden');
    }

    // Initial state
    console.log('Frontend initialized');
});