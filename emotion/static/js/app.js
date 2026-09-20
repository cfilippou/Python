const imageInput = document.getElementById("imageInput");
const inputPreview = document.getElementById("inputPreview");
const analyzeButton = document.getElementById("analyzeButton");
const statusText = document.getElementById("statusText");
const chatWindow = document.getElementById("chatWindow");





const refreshChartButton = document.getElementById("refreshChartButton");
const emotionChartCanvas = document.getElementById("emotionChart");
const loadingOverlay = document.getElementById("loadingOverlay");
const exportResultsButton = document.getElementById("exportResultsButton");
const clearResultsButton = document.getElementById("clearResultsButton");
const savedResultsCount = document.getElementById("savedResultsCount");
const exportStatusText = document.getElementById("exportStatusText");

let selectedFile = null;

let emotionChart = null;

const emotionTranslations = {
    happy: "χαρούμενος",
    sad: "λυπημένος",
    angry: "θυμωμένος",
    fear: "φοβισμένος",
    disgust: "αηδιασμένος",
    surprise: "έκπληκτος",
    neutral: "ουδέτερος",
    unknown: "άγνωστο"
};

function translateEmotion(label) {
    return emotionTranslations[String(label).toLowerCase()] || label;
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function setLoading(isLoading) {
    if (!loadingOverlay) return;
    loadingOverlay.classList.toggle("hidden", !isLoading);
}

function friendlyErrorMessage(data, fallback) {
    const type = data?.error_type;

    const messages = {
        no_face:
            "Δεν εντοπίστηκε πρόσωπο. Δοκίμασε άλλη εικόνα με καθαρό και ορατό πρόσωπο.",
        invalid_input:
            data?.error || "Η εικόνα που επιλέχθηκε δεν είναι έγκυρη.",
        emotion_model_error:
            "Παρουσιάστηκε πρόβλημα στο μοντέλο αναγνώρισης συναισθήματος.",
        output_error:
            "Η ανάλυση ολοκληρώθηκε, αλλά δεν ήταν δυνατή η αποθήκευση της εικόνας.",
        unexpected_error:
            "Παρουσιάστηκε μη αναμενόμενο σφάλμα. Δοκίμασε ξανά."
    };

    return messages[type] || data?.error || fallback;
}

function setStatus(message, isError = false) {
    statusText.textContent = message;
    statusText.classList.toggle("error", isError);
}

function addMessage(role, html) {
    const article = document.createElement("article");
    article.className = `message ${role}`;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = html;

    article.appendChild(bubble);
    chatWindow.appendChild(article);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function metricLabel(score) {
    const value = Number(score || 0);
    if (value >= 85) return "Excellent";
    if (value >= 70) return "Good";
    if (value >= 55) return "Moderate";
    return "Needs improvement";
}

function createSinglePromptTable(styleKey, result) {
    if (!result || !result.evaluation) {
        return "";
    }

    const styleNames = {
        strict: "Strict factual",
        structured: "Structured analytical",
        natural: "Natural descriptive"
    };

    const evaluation = result.evaluation;
    const rows = [
        ["Clarity", evaluation.clarity],
        ["Coherence", evaluation.coherence],
        ["Factual Accuracy", evaluation.factual_accuracy]
    ];

    return `
        <section class="prompt-result-card prompt-${escapeHtml(styleKey)}">
            <div class="prompt-result-header">
                <div>
                    <span class="prompt-index-label">
                        ${escapeHtml(styleNames[styleKey] || styleKey)}
                    </span>
                    <h3>Prompt Evaluation</h3>
                </div>
                <span class="overall-score">
                    Overall: ${Number(evaluation.overall_score).toFixed(1)}/100
                </span>
            </div>

            <p class="prompt-generated-text">
                ${escapeHtml(result.description || "")}
            </p>

            <div class="evaluation-table-wrap">
                <table class="evaluation-table">
                    <thead>
                        <tr>
                            <th>Criterion</th>
                            <th>Metric</th>
                            <th>Score</th>
                            <th>Assessment</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${rows.map(([criterion, metric]) => `
                            <tr>
                                <td>${criterion}</td>
                                <td>${escapeHtml(metric.metric)}</td>
                                <td>
                                    <strong>${Number(metric.score).toFixed(1)}/100</strong>
                                </td>
                                <td>${metricLabel(metric.score)}</td>
                            </tr>
                        `).join("")}
                    </tbody>
                </table>
            </div>
        </section>
    `;
}

function createPromptEngineeringResults(promptResults) {
    if (!promptResults) {
        return "";
    }

    const order = ["strict", "structured", "natural"];

    return `
        <div class="prompt-engineering-results">
            <div class="prompt-engineering-heading">
                <div>
                    <h2>Prompt Engineering Comparison</h2>
                    <p>
                        Η ίδια εικόνα αναλύθηκε αυτόματα με τρία διαφορετικά prompt styles.
                    </p>
                </div>
            </div>

            <div class="prompt-tables-grid">
                ${order.map(style =>
                    createSinglePromptTable(style, promptResults[style])
                ).join("")}
            </div>
        </div>
    `;
}


function createMusicRecommendations(dominantEmotion, recommendations) {
    if (!recommendations || recommendations.length === 0) {
        return "";
    }

    const emotionLabel = translateEmotion(dominantEmotion);

    return `
        <div class="music-recommendations">
            <h3>Προτεινόμενα τραγούδια</h3>
            <p class="music-context">
                Με βάση το dominant facial expression:
                <strong>${escapeHtml(emotionLabel)}</strong>
            </p>
            <div class="music-list">
                ${recommendations.map(item => `
                    <a
                        class="music-item"
                        href="${item.youtube_url}"
                        target="_blank"
                        rel="noopener noreferrer"
                        title="Άνοιγμα αναζήτησης στο YouTube"
                    >
                        <span>
                            <strong>${escapeHtml(item.artist)}</strong>,
                            ${escapeHtml(item.song)}
                        </span>
                        <span class="youtube-label">YouTube</span>
                    </a>
                `).join("")}
            </div>
        </div>
    `;
}

function emotionColor(label) {
    const colors = {
        angry: "#ef4444",
        disgust: "#f97316",
        fear: "#6d4aff",
        happy: "#2fb344",
        sad: "#2f80ed",
        surprise: "#f4b000",
        neutral: "#a8b0bb"
    };
    return colors[String(label).toLowerCase()] || "#6b7280";
}

function createEmotionDetails(emotions) {
    if (!emotions || emotions.length === 0) {
        return "<p>Δεν υπάρχει αξιόπιστο αποτέλεσμα.</p>";
    }

    const preferredOrder = [
        "angry", "disgust", "fear", "happy",
        "sad", "surprise", "neutral"
    ];

    return emotions.map((face, index) => {
        const scores = face.scores || {};
        const dominant = String(face.label || "").toLowerCase();
        const dominantConfidence = (Number(face.confidence) * 100).toFixed(1);

        const rows = preferredOrder
            .filter(label => Object.prototype.hasOwnProperty.call(scores, label))
            .map(label => {
                const rawScore = Number(scores[label] || 0);
                const boundedScore = Math.max(0, Math.min(100, rawScore));
                const isDominant = label === dominant;
                const color = emotionColor(label);

                return `
                    <div class="emotion-score-row ${isDominant ? "dominant" : ""}">
                        <span class="emotion-dot" style="background:${color}"></span>
                        <span class="emotion-score-name">${escapeHtml(label)}</span>
                        <div class="emotion-score-track">
                            <div class="emotion-score-fill" style="width:${boundedScore.toFixed(2)}%; background:${color}"></div>
                        </div>
                        <span class="emotion-score-value">${rawScore.toFixed(1)}%</span>
                    </div>
                `;
            }).join("");

        return `
            <div class="face-emotion-block">
                <div class="face-heading-row">
                    <div>
                        <h3>Face ${index + 1}</h3>
                        <p class="dominant-emotion-line">
                            Dominant Emotion:
                            <strong style="color:${emotionColor(dominant)}">
                                ${escapeHtml(dominant)} (${dominantConfidence}%)
                            </strong>
                        </p>
                    </div>
                </div>
                <div class="emotion-score-list">
                    ${rows}
                </div>
            </div>
        `;
    }).join("");
}


function createResultList(items, kind) {
    if (!items || items.length === 0) {
        return "<p>Δεν υπάρχει αξιόπιστο αποτέλεσμα.</p>";
    }

    return `
        <ul>
            ${items.map(item => `
                <li>
                    ${escapeHtml(kind === "emotion" ? translateEmotion(item.label) : item.label)}
                    — ${(Number(item.confidence) * 100).toFixed(1)}%
                </li>
            `).join("")}
        </ul>
    `;
}

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    selectedFile = file || null;

    if (!file) {
        inputPreview.classList.add("hidden");
        analyzeButton.disabled = true;
        return;
    }

    const url = URL.createObjectURL(file);
    inputPreview.src = url;
    inputPreview.classList.remove("hidden");
    analyzeButton.disabled = false;
    setStatus("Η εικόνα είναι έτοιμη για ανάλυση.");
});





analyzeButton.addEventListener("click", async () => {
    if (!selectedFile) {
        setStatus("Επίλεξε πρώτα μια εικόνα.", true);
        return;
    }

    analyzeButton.disabled = true;
    setLoading(true);
    setStatus("Γίνεται αναγνώριση συναισθήματος, ανίχνευση αντικειμένων και δημιουργία περιγραφής από το LLM...");

    const formData = new FormData();

    formData.append("image", selectedFile);

    const previewSource = URL.createObjectURL(selectedFile);
    addMessage("user", `<img class="result-image" src="${previewSource}" alt="Εικόνα που υποβλήθηκε">`);

    try {
        const response = await fetch("/analyze", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || !data.ok) {
            throw new Error(friendlyErrorMessage(data, "Η ανάλυση απέτυχε."));
        }

        addMessage("assistant", `
            <div class="analysis-result-layout">
                <section class="analysis-panel face-panel">
                    <div class="panel-title-row">
                        <h2>Detected Faces (${data.emotions?.length || 0})</h2>
                    </div>

                    <img
                        class="result-image"
                        src="${data.annotated_image_url}"
                        alt="Annotated facial emotion analysis"
                    >

                    <div class="faces-details">
                        ${createEmotionDetails(data.emotions)}
                    </div>
                </section>

                <section class="analysis-panel description-panel">
                    <div class="panel-title-row">
                        <h2>AI Descriptions</h2>
                    </div>

                    <div class="prompt-description-list">
                        <article class="prompt-description-card">
                            <div class="prompt-description-header">
                                <h3>Strict factual</h3>
                                <span class="primary-prompt-badge">Prompt 1</span>
                            </div>
                            <p class="description-text">
                                ${escapeHtml(data.prompt_results?.strict?.description || data.description || "")}
                            </p>
                        </article>

                        <article class="prompt-description-card">
                            <div class="prompt-description-header">
                                <h3>Structured analytical</h3>
                                <span class="primary-prompt-badge">Prompt 2</span>
                            </div>
                            <p class="description-text">
                                ${escapeHtml(data.prompt_results?.structured?.description || "")}
                            </p>
                        </article>

                        <article class="prompt-description-card">
                            <div class="prompt-description-header">
                                <h3>Natural descriptive</h3>
                                <span class="primary-prompt-badge">Prompt 3</span>
                            </div>
                            <p class="description-text">
                                ${escapeHtml(data.prompt_results?.natural?.description || "")}
                            </p>
                        </article>
                    </div>
                </section>
            </div>

            ${createPromptEngineeringResults(data.prompt_results)}

            ${createMusicRecommendations(
                data.dominant_emotion,
                data.music_recommendations
            )}
        `);

        setStatus("Η ανάλυση ολοκληρώθηκε.");
        await loadResultsStatus();
        await loadEmotionChart();

    } catch (error) {
        addMessage(
            "assistant",
            `<p>Σφάλμα ανάλυσης: ${escapeHtml(error.message)}</p>`
        );
        setStatus(error.message, true);
    } finally {
        setLoading(false);
        analyzeButton.disabled = false;
    }
});

refreshChartButton.addEventListener("click", loadEmotionChart);

async function loadEmotionChart() {
    try {
        const response = await fetch("/emotion-summary");
        const data = await response.json();
        const counts = data.counts || {};

        const labels = Object.keys(counts);
        const values = Object.values(counts);

        if (emotionChart) {
            emotionChart.destroy();
        }

        emotionChart = new Chart(emotionChartCanvas, {
            type: "bar",
            data: {
                labels,
                datasets: [{
                    label: "Πλήθος ανιχνεύσεων συναισθήματος",
                    data: values
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0
                        }
                    }
                }
            }
        });
    } catch (error) {
        setStatus("Δεν ήταν δυνατή η φόρτωση του γραφήματος συναισθημάτων.", true);
    }
}

loadEmotionChart();


async function loadResultsStatus() {
    if (!savedResultsCount) return;

    try {
        const response = await fetch("/results-status");
        const data = await response.json();
        savedResultsCount.textContent = Number(data.saved_rows || 0);
    } catch (error) {
        savedResultsCount.textContent = "—";
    }
}

if (exportResultsButton) {
    exportResultsButton.addEventListener("click", async () => {
        if (exportStatusText) {
            exportStatusText.textContent = "Προετοιμασία αρχείου CSV...";
            exportStatusText.classList.remove("error");
        }

        try {
            const response = await fetch("/results-status");
            const data = await response.json();

            if (!data.saved_rows) {
                throw new Error("Δεν υπάρχουν ακόμη αποθηκευμένα αποτελέσματα.");
            }

            window.location.href = "/export-results";

            if (exportStatusText) {
                exportStatusText.textContent = "Το CSV είναι έτοιμο για λήψη.";
            }
        } catch (error) {
            if (exportStatusText) {
                exportStatusText.textContent = error.message;
                exportStatusText.classList.add("error");
            }
        }
    });
}

if (clearResultsButton) {
    clearResultsButton.addEventListener("click", async () => {
        const confirmed = window.confirm(
            "Να διαγραφούν όλα τα αποθηκευμένα πειραματικά αποτελέσματα;"
        );
        if (!confirmed) return;

        try {
            const response = await fetch("/clear-results", {method: "POST"});
            const data = await response.json();

            if (!response.ok || !data.ok) {
                throw new Error(data.error || "Η διαγραφή απέτυχε.");
            }

            if (exportStatusText) {
                exportStatusText.textContent =
                    "Τα αποθηκευμένα αποτελέσματα διαγράφηκαν.";
                exportStatusText.classList.remove("error");
            }

            await loadResultsStatus();
        } catch (error) {
            if (exportStatusText) {
                exportStatusText.textContent = error.message;
                exportStatusText.classList.add("error");
            }
        }
    });
}

loadResultsStatus();
