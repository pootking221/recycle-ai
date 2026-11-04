
import * as ort from "onnxruntime-web";

// ==========================================
// 1. กำหนดค่าพื้นฐานของโมเดล
// ==========================================

// รายชื่อคลาสที่โมเดลสามารถตรวจจับได้
const CLASS_NAMES = ["Metal", "Paper", "Plastic"];

// กำหนดสี RGB สำหรับแสดงผลวัตถุแต่ละประเภท
// Metal = สีแดง, Paper = สีน้ำเงิน, Plastic = สีเหลือง
const CLASS_COLORS = [
    [239, 68, 68],
    [59, 130, 246],
    [234, 179, 8],
];


// ==========================================
// 2. โหลดโมเดล ONNX
// ==========================================

export async function loadModel(modelPath) {

    // สร้าง Inference Session เพื่อเตรียมโมเดลสำหรับประมวลผล
    // modelPath คือที่อยู่ของไฟล์โมเดล ONNX
    // ใช้ WASM เพื่อให้โมเดลทำงานภายใน Web Browser
    const session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ["wasm"],
    });

    
    console.log("Model loaded:", modelPath);

    // ส่ง Session กลับไปให้หน้าเว็บเรียกใช้ประมวลผล
    return session;
}


// ==========================================
// 3. เตรียมภาพก่อนส่งเข้าโมเดล
// ==========================================

export async function preprocessImage(image) {

    // กำหนดขนาดภาพที่โมเดลต้องการ คือ 640x640 พิกเซล
    const size = 640;

    // สร้าง Canvas สำหรับปรับขนาดภาพ
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;

    // ดึง Context สำหรับจัดการภาพบน Canvas
    const ctx = canvas.getContext("2d");

    // วาดภาพต้นฉบับและปรับขนาดเป็น 640x640
    ctx.drawImage(image, 0, 0, size, size);

    // อ่านข้อมูลพิกเซลของภาพในรูปแบบ RGBA
    const { data } = ctx.getImageData(0, 0, size, size);

    // สร้าง Array สำหรับเก็บข้อมูลภาพในรูปแบบ Float32
    // ข้อมูลถูกแบ่งออกเป็น 3 ช่อง ได้แก่ Red, Green และ Blue
    const input = new Float32Array(3 * size * size);

    // วนอ่านค่าพิกเซลของภาพทั้งหมด
    for (let i = 0; i < size * size; i++) {

        // แยกช่องสี Red และปรับค่าให้อยู่ระหว่าง 0-1
        input[i] = data[i * 4] / 255;

        // แยกช่องสี Green และปรับค่าให้อยู่ระหว่าง 0-1
        input[i + size * size] = data[i * 4 + 1] / 255;

        // แยกช่องสี Blue และปรับค่าให้อยู่ระหว่าง 0-1
        input[i + size * size * 2] = data[i * 4 + 2] / 255;
    }

    // แปลงข้อมูลภาพเป็น Tensor สำหรับป้อนเข้าโมเดล
    // รูปแบบข้อมูลคือ [Batch, Channel, Height, Width]
    // [1, 3, 640, 640] หมายถึง 1 ภาพ มี 3 ช่องสี ขนาด 640x640
    return new ort.Tensor("float32", input, [1, 3, size, size]);
}


// ==========================================
//  ใช้คำนวณพื้นที่ที่กรอบวัตถุสองกรอบซ้อนทับกัน
// ==========================================

function calculateIoU(a, b) {

    // หาพิกัดขอบเขตที่ซ้อนทับกันของกรอบทั้งสอง
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);

    // คำนวณพื้นที่ส่วนที่กรอบทั้งสองซ้อนทับกัน
    // หากไม่มีพื้นที่ซ้อนทับ จะได้ค่าเป็น 0
    const intersection =
        Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

    // คำนวณพื้นที่ทั้งหมดของกรอบวัตถุ A
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);

    // คำนวณพื้นที่ทั้งหมดของกรอบวัตถุ B
    const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

    // คำนวณ IoU โดยใช้พื้นที่ซ้อนทับหารด้วยพื้นที่รวม
    return intersection / (areaA + areaB - intersection);
}


// ==========================================
// 5. คัดกรองกรอบวัตถุด้วย NMS
// ==========================================

// Non-Maximum Suppression (NMS)
// ใช้ลบกรอบตรวจจับที่ซ้ำกัน โดยเลือกกรอบที่มีความมั่นใจสูงกว่า
function applyNMS(detections, threshold = 0.45) {

    // สร้าง Array สำหรับเก็บกรอบวัตถุที่ผ่านการคัดกรอง
    const selected = [];

    // เรียงกรอบวัตถุตามค่าความมั่นใจจากมากไปน้อย
    detections.sort((a, b) => b.confidence - a.confidence);

    // ทำงานจนกว่าจะตรวจสอบกรอบวัตถุครบทั้งหมด
    while (detections.length) {

        // เลือกกรอบที่มีค่าความมั่นใจสูงที่สุด
        const best = detections.shift();

        // เก็บกรอบที่เลือกไว้ในผลลัพธ์
        selected.push(best);

        // ตรวจสอบกรอบที่เหลือว่าซ้ำกับกรอบที่เลือกหรือไม่
        for (let i = detections.length - 1; i >= 0; i--) {

            // เปรียบเทียบเฉพาะกรอบที่เป็นคลาสเดียวกัน
            // หากค่า IoU มากกว่า 0.45 จะถือว่ากรอบซ้ำกัน
            if (
                detections[i].classId === best.classId &&
                calculateIoU(best, detections[i]) > threshold
            ) {

                // ลบกรอบที่ซ้ำออกจากรายการ
                detections.splice(i, 1);
            }
        }
    }

    // ส่งคืนกรอบวัตถุที่ผ่านการคัดกรองแล้ว
    return selected;
}


// ==========================================
// 6. อ่านผลลัพธ์จากโมเดล YOLO
// ==========================================

// ใช้ฟังก์ชันเดียวกันในการอ่านผลลัพธ์ของ Detection และ Segmentation
function readPredictions(output, confidenceThreshold = 0.25, numMasks = 0) {

    // ดึงข้อมูลตัวเลขที่โมเดลทำนายได้
    const predictions = output.data;

    // อ่านจำนวนตำแหน่งที่โมเดลใช้ตรวจจับวัตถุ
    const numPredictions = output.dims[2];

    // สร้าง Array สำหรับเก็บผลการตรวจจับ
    const detections = [];

    // วนตรวจสอบผลลัพธ์ของแต่ละตำแหน่ง
    for (let i = 0; i < numPredictions; i++) {

        // กำหนดค่าเริ่มต้นของคลาสและความมั่นใจ
        let classId = 0;
        let confidence = 0;

        // ตรวจสอบค่าความมั่นใจของทั้ง 3 คลาส
        for (let c = 0; c < CLASS_NAMES.length; c++) {

            // อ่านค่าความมั่นใจของคลาสปัจจุบัน
            const score = predictions[(4 + c) * numPredictions + i];

            // หากค่าความมั่นใจสูงกว่าเดิม ให้เลือกคลาสนี้
            if (score > confidence) {
                confidence = score;
                classId = c;
            }
        }

        // ข้ามผลการตรวจจับที่มีความมั่นใจต่ำกว่าเกณฑ์
        // ค่าเริ่มต้นคือ 0.25 หรือ 25%
        if (confidence < confidenceThreshold) continue;

        // อ่านพิกัดจุดกึ่งกลางของกรอบวัตถุ
        const cx = predictions[i];
        const cy = predictions[numPredictions + i];

        // อ่านความกว้างและความสูงของกรอบวัตถุ
        const w = predictions[2 * numPredictions + i];
        const h = predictions[3 * numPredictions + i];

        // แปลงพิกัดจากจุดกึ่งกลางเป็นพิกัดมุมทั้งสี่
        const detection = {
            classId,
            className: CLASS_NAMES[classId],
            confidence,

            // พิกัดมุมซ้ายบน
            x1: cx - w / 2,
            y1: cy - h / 2,

            // พิกัดมุมขวาล่าง
            x2: cx + w / 2,
            y2: cy + h / 2,
        };

        // หากเป็น Segmentation จะต้องอ่านค่า Mask Coefficients เพิ่มเติม
        if (numMasks > 0) {

            // สร้าง Array สำหรับเก็บค่า Coefficients ทั้ง 32 ค่า
            detection.coefficients = [];

            // วนอ่านค่า Coefficients จากผลลัพธ์ของโมเดล
            for (let m = 0; m < numMasks; m++) {
                detection.coefficients.push(
                    predictions[(7 + m) * numPredictions + i]
                );
            }
        }

        // เพิ่มผลการตรวจจับลงใน Array
        detections.push(detection);
    }

    // ส่งคืนข้อมูลวัตถุที่ผ่านการตรวจสอบ Confidence แล้ว
    return detections;
}


// ==========================================
// 7. ประมวลผล Object Detection
// ==========================================

export function processDetection(output, confidenceThreshold = 0.25) {

    // อ่านผลลัพธ์และคัดกรองวัตถุตามค่า Confidence
    const detections = readPredictions(output, confidenceThreshold);

    // ใช้ NMS เพื่อลบกรอบวัตถุที่ตรวจจับซ้ำ
    return applyNMS(detections);
}


// ==========================================
// 8. ประมวลผล Segmentation
// ==========================================

export function processSegmentation(output0, output1, confidenceThreshold = 0.25) {

    // อ่านผลลัพธ์ของโมเดล Segmentation
    // กำหนดให้มี Mask Coefficients จำนวน 32 ค่า
    const detections = readPredictions(output0, confidenceThreshold, 32);

    // คัดกรองกรอบที่ซ้ำกันด้วย NMS
    // จากนั้นเปลี่ยนชื่อ coefficients เป็น maskCoefficients
    return applyNMS(detections).map(item => ({
        ...item,
        maskCoefficients: item.coefficients,
    }));
}


// ==========================================
// 9. สร้างภาพ Mask Overlay สำหรับ Segmentation
// ==========================================




 // สร้าง Segmentation Mask และแสดงสีบนวัตถุที่ตรวจพบ
export function createSegmentationOverlay(
    output0,
    output1,
    imageWidth,
    imageHeight,
    confidenceThreshold = 0.25,
    maskThreshold = 0.5
) {
    // อ่านผลลัพธ์และกรองกรอบที่ซ้ำกัน
    const detections = applyNMS(
        readPredictions(output0, confidenceThreshold, 32)
    );

    // ดึงข้อมูล Prototype Mask จากโมเดล
    const prototypes = output1.data;
    const numMasks = output1.dims[1];
    const maskHeight = output1.dims[2];
    const maskWidth = output1.dims[3];

    // สร้าง Canvas สำหรับแสดงผล Mask
    const canvas = document.createElement("canvas");
    canvas.width = imageWidth;
    canvas.height = imageHeight;

    const ctx = canvas.getContext("2d");

    // สร้างพื้นที่ข้อมูลสีและความโปร่งใส
    const imageData = ctx.createImageData(imageWidth, imageHeight);

    // วนลูปวัตถุที่ตรวจพบแต่ละรายการ
    for (const detection of detections) {

        // กำหนดสีตามประเภทของวัตถุ
        const color = CLASS_COLORS[detection.classId];

        // วนลูปพิกัดทุก Pixel ของภาพต้นฉบับ
        for (let y = 0; y < imageHeight; y++) {
            for (let x = 0; x < imageWidth; x++) {

                // แปลงพิกัดภาพต้นฉบับเป็นขนาด 640x640
                const modelX = (x / imageWidth) * 640;
                const modelY = (y / imageHeight) * 640;

                // ตรวจสอบว่า Pixel อยู่ภายในกรอบวัตถุหรือไม่
                if (
                    modelX < detection.x1 ||
                    modelX > detection.x2 ||
                    modelY < detection.y1 ||
                    modelY > detection.y2
                ) continue;

                // แปลงพิกัดเป็นตำแหน่งใน Prototype Mask
                const maskX = Math.min(
                    maskWidth - 1,
                    Math.floor((modelX / 640) * maskWidth)
                );

                const maskY = Math.min(
                    maskHeight - 1,
                    Math.floor((modelY / 640) * maskHeight)
                );

                // คำนวณค่าของ Mask จาก Coefficients และ Prototypes
                let value = 0;

                for (let m = 0; m < numMasks; m++) {
                    const index =
                        m * maskHeight * maskWidth +
                        maskY * maskWidth +
                        maskX;

                    value +=
                        detection.coefficients[m] * prototypes[index];
                }

                // แปลงค่า Mask ด้วย Sigmoid ให้อยู่ในช่วง 0-1
                const probability = 1 / (1 + Math.exp(-value));

                // ตรวจสอบว่า Pixel ผ่านค่า Mask Threshold หรือไม่
                if (probability >= maskThreshold) {

                    // คำนวณตำแหน่ง Pixel ในภาพต้นฉบับ
                    const index = (y * imageWidth + x) * 4;

                    // กำหนดสี RGB ตามประเภทวัตถุ
                    imageData.data[index] = color[0];
                    imageData.data[index + 1] = color[1];
                    imageData.data[index + 2] = color[2];

                    // กำหนดความโปร่งใสของ Mask
                    imageData.data[index + 3] = 150;
                }
            }
        }
    }

    // นำข้อมูลสี Mask ไปแสดงบน Canvas
    ctx.putImageData(imageData, 0, 0);

    // ส่งคืนภาพ Mask จำนวนวัตถุ และรายละเอียดผลการตรวจจับ
    return {
        image: canvas.toDataURL("image/png"),
        count: detections.length,
        detections,
    };
}


