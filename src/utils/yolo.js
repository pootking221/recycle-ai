import * as ort from "onnxruntime-web";

export async function loadModel(modelPath) {
    const session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ["wasm"],
    });

    console.log("Model loaded:", modelPath);

    return session;
}

// แปลงภาพเป็น Tensor
export async function preprocessImage(image) {
    const size = 640;

    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;

    const ctx = canvas.getContext("2d");

    ctx.drawImage(image, 0, 0, size, size);

    const imageData = ctx.getImageData(0, 0, size, size);
    const pixels = imageData.data;

    const tensorData = new Float32Array(3 * size * size);

    for (let i = 0; i < size * size; i++) {
        tensorData[i] = pixels[i * 4] / 255;
        tensorData[i + size * size] = pixels[i * 4 + 1] / 255;
        tensorData[i + 2 * size * size] = pixels[i * 4 + 2] / 255;
    }

    return new ort.Tensor(
        "float32",
        tensorData,
        [1, 3, size, size]
    );
}

// แปลผล Output จาก YOLO Detection
export function processDetection(output, confidenceThreshold = 0.25) {
    const data = output.data;

    const numPredictions = output.dims[2];
    const numClasses = 3;

    const classNames = ["Metal", "Paper", "Plastic"];

    const boxes = [];

    for (let i = 0; i < numPredictions; i++) {
        // อ่านคะแนนแต่ละ Class
        let maxScore = 0;
        let classId = -1;

        for (let c = 0; c < numClasses; c++) {
            const score = data[(4 + c) * numPredictions + i];

            if (score > maxScore) {
                maxScore = score;
                classId = c;
            }
        }

        if (maxScore < confidenceThreshold) continue;

        // อ่านพิกัดกรอบ (Center X, Center Y, Width, Height)
        const cx = data[i];
        const cy = data[numPredictions + i];
        const w = data[2 * numPredictions + i];
        const h = data[3 * numPredictions + i];

        boxes.push({
            x1: cx - w / 2,
            y1: cy - h / 2,
            x2: cx + w / 2,
            y2: cy + h / 2,
            score: maxScore,
            classId,
            className: classNames[classId],
        });
    }

    // Non-Maximum Suppression
    boxes.sort((a, b) => b.score - a.score);

    const selected = [];

    while (boxes.length > 0) {
        const best = boxes.shift();
        selected.push(best);

        for (let i = boxes.length - 1; i >= 0; i--) {
            if (boxes[i].classId === best.classId) {
                const overlap = calculateIoU(best, boxes[i]);

                if (overlap > 0.45) {
                    boxes.splice(i, 1);
                }
            }
        }
    }

    return selected;
}

// คำนวณพื้นที่ซ้อนทับของกรอบ
function calculateIoU(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);

    const intersection = Math.max(0, x2 - x1) *
        Math.max(0, y2 - y1);

    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

    return intersection / (areaA + areaB - intersection + 1e-6);
}

// ประมวลผล Output ของ YOLO Segmentation
export function processSegmentation(output0, output1, confidenceThreshold = 0.25) {
    const predictions = output0.data;
    const prototypes = output1.data;

    const numPredictions = 8400;
    const numClasses = 3;
    const numMasks = 32;
    const maskSize = 160;

    const classNames = ["Metal", "Paper", "Plastic"];

    const detections = [];

    for (let i = 0; i < numPredictions; i++) {
        let maxScore = 0;
        let classId = -1;

        // อ่านคะแนน Class
        for (let c = 0; c < numClasses; c++) {
            const score = predictions[(4 + c) * numPredictions + i];

            if (score > maxScore) {
                maxScore = score;
                classId = c;
            }
        }

        if (maxScore < confidenceThreshold) continue;

        // อ่าน Bounding Box
        const cx = predictions[i];
        const cy = predictions[numPredictions + i];
        const w = predictions[2 * numPredictions + i];
        const h = predictions[3 * numPredictions + i];

        // อ่าน Mask Coefficients
        const maskCoefficients = [];

        for (let m = 0; m < numMasks; m++) {
            maskCoefficients.push(
                predictions[(7 + m) * numPredictions + i]
            );
        }

        detections.push({
            x1: cx - w / 2,
            y1: cy - h / 2,
            x2: cx + w / 2,
            y2: cy + h / 2,
            score: maxScore,
            classId,
            className: classNames[classId],
            maskCoefficients,
        });
    }

    return detections;
}


// สร้าง Mask Overlay พร้อมกรอบและชื่อประเภทขยะ
export function createSegmentationOverlay(
    output0,
    output1,
    imageWidth,
    imageHeight,
    confidenceThreshold = 0.25,
    maskThreshold = 0.5
) {
    const predictions = output0.data;
    const prototypes = output1.data;

    const numPredictions = output0.dims[2];
    const numClasses = 3;
    const numMasks = 32;

    const maskHeight = output1.dims[2];
    const maskWidth = output1.dims[3];
    const maskArea = maskHeight * maskWidth;

    // ชื่อประเภทขยะตามลำดับ Class ID
    const classNames = ["Metal", "Paper", "Plastic"];

    // สีของแต่ละประเภท
    const classColors = [
        [239, 68, 68],   // Metal - Red
        [59, 130, 246],  // Paper - Blue
        [234, 179, 8],   // Plastic - Yellow
    ];

    const detections = [];

    // อ่านผลลัพธ์จากโมเดล
    for (let i = 0; i < numPredictions; i++) {
        let maxScore = 0;
        let classId = -1;

        for (let c = 0; c < numClasses; c++) {
            const score =
                predictions[(4 + c) * numPredictions + i];

            if (score > maxScore) {
                maxScore = score;
                classId = c;
            }
        }

        if (maxScore < confidenceThreshold) continue;

        const cx = predictions[i];
        const cy = predictions[numPredictions + i];
        const w = predictions[2 * numPredictions + i];
        const h = predictions[3 * numPredictions + i];

        const coefficients = [];

        for (let m = 0; m < numMasks; m++) {
            coefficients.push(
                predictions[(7 + m) * numPredictions + i]
            );
        }

        detections.push({
            x1: cx - w / 2,
            y1: cy - h / 2,
            x2: cx + w / 2,
            y2: cy + h / 2,
            score: maxScore,
            classId,
            className: classNames[classId],
            coefficients,
        });
    }

    // เรียงตาม Confidence
    detections.sort((a, b) => b.score - a.score);

    // Non-Maximum Suppression
    const selected = [];

    function calculateIoU(a, b) {
        const x1 = Math.max(a.x1, b.x1);
        const y1 = Math.max(a.y1, b.y1);
        const x2 = Math.min(a.x2, b.x2);
        const y2 = Math.min(a.y2, b.y2);

        const intersection =
            Math.max(0, x2 - x1) *
            Math.max(0, y2 - y1);

        const areaA =
            Math.max(0, a.x2 - a.x1) *
            Math.max(0, a.y2 - a.y1);

        const areaB =
            Math.max(0, b.x2 - b.x1) *
            Math.max(0, b.y2 - b.y1);

        return intersection /
            (areaA + areaB - intersection + 1e-6);
    }

    while (detections.length > 0) {
        const best = detections.shift();
        selected.push(best);

        for (let i = detections.length - 1; i >= 0; i--) {
            if (
                detections[i].classId === best.classId &&
                calculateIoU(best, detections[i]) > 0.45
            ) {
                detections.splice(i, 1);
            }
        }
    }

    // สร้าง Canvas
    const canvas = document.createElement("canvas");
    canvas.width = imageWidth;
    canvas.height = imageHeight;

    const ctx = canvas.getContext("2d");

    const imageData = ctx.createImageData(
        imageWidth,
        imageHeight
    );

    // สร้าง Mask ของแต่ละวัตถุ
    for (const detection of selected) {
        const { coefficients, classId } = detection;
        const color = classColors[classId];

        for (let y = 0; y < maskHeight; y++) {
            for (let x = 0; x < maskWidth; x++) {
                const px = y * maskWidth + x;

                const modelX =
                    (x + 0.5) * 640 / maskWidth;

                const modelY =
                    (y + 0.5) * 640 / maskHeight;

                // ตัดพื้นที่นอกกรอบ
                if (
                    modelX < detection.x1 ||
                    modelX > detection.x2 ||
                    modelY < detection.y1 ||
                    modelY > detection.y2
                ) continue;

                // คำนวณ Mask จาก 32 Prototypes
                let value = 0;

                for (let m = 0; m < numMasks; m++) {
                    value +=
                        coefficients[m] *
                        prototypes[m * maskArea + px];
                }

                const probability =
                    1 / (1 + Math.exp(-value));

                if (probability < maskThreshold) continue;

                // แปลงพิกัดเป็นภาพจริง
                const startX = Math.floor(
                    x * imageWidth / maskWidth
                );

                const endX = Math.floor(
                    (x + 1) * imageWidth / maskWidth
                );

                const startY = Math.floor(
                    y * imageHeight / maskHeight
                );

                const endY = Math.floor(
                    (y + 1) * imageHeight / maskHeight
                );

                for (let py = startY; py < endY; py++) {
                    for (let px = startX; px < endX; px++) {
                        const index =
                            (py * imageWidth + px) * 4;

                        imageData.data[index] = color[0];
                        imageData.data[index + 1] = color[1];
                        imageData.data[index + 2] = color[2];
                        imageData.data[index + 3] = 130;
                    }
                }
            }
        }
    }

    // วาด Mask ลงบน Canvas
    ctx.putImageData(imageData, 0, 0);

    // ปรับขนาดพิกัดกรอบจาก 640 เป็นขนาดภาพจริง
    const scaleX = imageWidth / 640;
    const scaleY = imageHeight / 640;

    // วาดกรอบและข้อความกำกับ
    selected.forEach((detection) => {
        const color = classColors[detection.classId];

        const x = Math.max(0, detection.x1 * scaleX);
        const y = Math.max(0, detection.y1 * scaleY);

        const right = Math.min(
            imageWidth,
            detection.x2 * scaleX
        );

        const bottom = Math.min(
            imageHeight,
            detection.y2 * scaleY
        );

        const width = right - x;
        const height = bottom - y;

        // วาดกรอบ Bounding Box
        ctx.strokeStyle =
            `rgb(${color[0]}, ${color[1]}, ${color[2]})`;

        ctx.lineWidth = Math.max(
            2,
            Math.min(imageWidth, imageHeight) * 0.004
        );

        

        // ข้อความชื่อประเภทและ Confidence
        const label =
            `${detection.className} ${(detection.score * 100).toFixed(2)}%`;

        const fontSize = Math.max(
            14,
            Math.min(24, imageWidth * 0.025)
        );

        ctx.font = `bold ${fontSize}px Arial`;

        const textWidth = ctx.measureText(label).width;

        const padding = 6;
        const labelHeight = fontSize + padding * 2;

        // พื้นหลังป้ายชื่อ
        const labelY = Math.max(0, y - labelHeight);

        ctx.fillStyle =
            `rgb(${color[0]}, ${color[1]}, ${color[2]})`;

        ctx.fillRect(
            x,
            labelY,
            textWidth + padding * 2,
            labelHeight
        );

        // ข้อความ
        ctx.fillStyle = "#ffffff";

        ctx.fillText(
            label,
            x + padding,
            labelY + fontSize + padding / 2
        );
    });

    // ส่งผลลัพธ์กลับไปยังหน้าเว็บ
    return {
        image: canvas.toDataURL("image/png"),
        count: selected.length,
        detections: selected,
    };
}
