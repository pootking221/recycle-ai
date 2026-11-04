
import { useRef, useState } from "react";
import {
    loadModel,
    preprocessImage,
    createSegmentationOverlay,
} from "../utils/yolo";

function Segmentation() {
    // เก็บข้อมูลภาพ กล้อง และผลลัพธ์
    const [image, setImage] = useState(null);
    const [camera, setCamera] = useState(false);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [maskImage, setMaskImage] = useState(null);

    const videoRef = useRef(null);
    const streamRef = useRef(null);

    // ล้างภาพและผลลัพธ์เดิม
    const clearPreviousImage = () => {
        setImage(null);
        setMaskImage(null);
        setResults(null);
    };

    // เปิดกล้องหลัง
    const openCamera = async () => {
        clearPreviousImage();

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "environment" },
            });

            streamRef.current = stream;
            setCamera(true);

            setTimeout(() => {
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                }
            }, 100);
        } catch (error) {
            console.error("Camera Error:", error);
            alert("ไม่สามารถเปิดกล้องได้");
        }
    };

    // ปิดกล้องและหยุด Stream
    const closeCamera = () => {
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        setCamera(false);
    };

    // ถ่ายภาพจากกล้อง
    const takePhoto = () => {
        const video = videoRef.current;

        if (!video || !video.videoWidth) {
            alert("กล้องยังไม่พร้อม");
            return;
        }

        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0);

        setImage(canvas.toDataURL("image/jpeg"));
        setMaskImage(null);
        setResults(null);

        closeCamera();
    };

    // เลือกภาพจากเครื่อง
    const handleImageUpload = (event) => {
        const file = event.target.files?.[0];

        if (!file) return;

        if (!file.type.startsWith("image/")) {
            alert("กรุณาเลือกไฟล์รูปภาพ");
            event.target.value = "";
            return;
        }

        clearPreviousImage();
        setImage(URL.createObjectURL(file));
        event.target.value = "";
    };

    // ประมวลผลภาพด้วย YOLO Segmentation
    const runSegmentation = async () => {
        if (!image || loading) return;

        try {
            setLoading(true);
            setResults(null);
            setMaskImage(null);

            // โหลดโมเดล Segmentation
            console.log("กำลังโหลด Segmentation Model...");

            const session = await loadModel("/models/segmentation_1.onnx");

            // โหลดภาพและเตรียมข้อมูลก่อนเข้าโมเดล
            const img = new Image();
            img.src = image;

            await new Promise((resolve, reject) => {
                img.onload = resolve;
                img.onerror = reject;
            });

            const inputTensor = await preprocessImage(img);

            console.log("กำลังประมวลผลภาพ...");

            // ส่งภาพเข้าโมเดล
            const output = await session.run({
                images: inputTensor,
            });

            // แสดงขนาด Output ใน Console
            console.log("Segmentation Output 0:", output.output0.dims);
            console.log("Segmentation Output 1:", output.output1.dims);

            // สร้าง Mask จากผลลัพธ์ของโมเดล
            const result = createSegmentationOverlay(
                output.output0,
                output.output1,
                img.naturalWidth,
                img.naturalHeight
            );

            console.log("Segmentation Results:", result.detections);
            console.log("จำนวนวัตถุที่ตรวจพบ:", result.count);

            // แสดงภาพ Mask และผลการตรวจจับ
            setMaskImage(result.image);
            setResults(result.detections);

        } catch (error) {
            console.error("Segmentation Error:", error);
            alert("ไม่สามารถประมวลผล Segmentation ได้");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="space-y-5">

            {/* หัวข้อหน้าเว็บ */}
            <div>
                <h1 className="text-2xl font-bold text-gray-800">
                    Polygon Segmentation
                </h1>

                <p className="mt-1 text-gray-500">
                    ตรวจจับและแสดงขอบเขตขยะรีไซเคิลด้วย YOLO
                </p>

                {/* คำอธิบายสีของ Segmentation */}
                <div className="mb-3 flex flex-wrap gap-4 text-sm">
                    <div className="flex items-center gap-2">
                        <span className="h-4 w-4 rounded bg-red-500"></span>
                        <span>Metal (โลหะ)</span>
                    </div>

                    <div className="flex items-center gap-2">
                        <span className="h-4 w-4 rounded bg-blue-500"></span>
                        <span>Paper (กระดาษ)</span>
                    </div>

                    <div className="flex items-center gap-2">
                        <span className="h-4 w-4 rounded bg-yellow-400"></span>
                        <span>Plastic (พลาสติก)</span>
                    </div>
                </div>
            </div>

            {/* ส่วนเลือกภาพและแสดงผล */}
            <div className="rounded-xl border bg-white p-6">

                <h2 className="mb-4 font-semibold text-gray-800">
                    เลือกภาพสำหรับประมวลผล
                </h2>

                {/* อัปโหลดภาพหรือเปิดกล้อง */}
                <div className="flex flex-wrap gap-3">

                    <label className="cursor-pointer rounded-lg bg-green-700 px-4 py-2 text-white hover:bg-green-800">
                        Upload Image

                        <input
                            type="file"
                            accept="image/*"
                            className="hidden"
                            onChange={handleImageUpload}
                        />
                    </label>

                    <button
                        onClick={openCamera}
                        className="rounded-lg border border-green-700 px-4 py-2 text-green-700 hover:bg-green-50"
                    >
                        Open Camera
                    </button>
                </div>

                {/* แสดงภาพจากกล้อง */}
                {camera && (
                    <div className="mt-5 space-y-3">
                        <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            className="max-h-96 w-full rounded-lg bg-black object-contain"
                        />

                        <div className="flex gap-3">
                            <button
                                onClick={takePhoto}
                                className="rounded-lg bg-green-700 px-4 py-2 text-white"
                            >
                                Take Photo
                            </button>

                            <button
                                onClick={closeCamera}
                                className="rounded-lg bg-gray-200 px-4 py-2"
                            >
                                Close Camera
                            </button>
                        </div>
                    </div>
                )}

                {/* แสดงภาพต้นฉบับและ Segmentation Mask */}
                {image && (
                    <div className="mt-6">
                        <h2 className="mb-3 font-semibold text-gray-800">
                            Segmentation Preview
                        </h2>

                        <div className="relative inline-block max-w-full overflow-hidden rounded-lg bg-gray-100">

                            <img
                                src={image}
                                alt="Original Image"
                                className="block max-h-96 max-w-full"
                            />

                            {maskImage && (
                                <img
                                    src={maskImage}
                                    alt="Segmentation Mask"
                                    className="pointer-events-none absolute inset-0 h-full w-full"
                                />
                            )}
                        </div>

                        <p className="mt-2 text-xs text-gray-400">
                            {maskImage
                                ? "แสดงภาพต้นฉบับพร้อม Segmentation Mask"
                                : "ภาพต้นฉบับ"}
                        </p>

                        {/* ลบภาพและผลลัพธ์ */}
                        <button
                            onClick={clearPreviousImage}
                            className="mt-3 rounded-lg bg-red-600 px-4 py-2 text-white hover:bg-red-700"
                        >
                            Remove Image
                        </button>
                    </div>
                )}

                {/* เริ่มประมวลผล Segmentation */}
                <button
                    onClick={runSegmentation}
                    disabled={!image || loading}
                    className="mt-5 rounded-lg bg-green-700 px-5 py-2 text-white hover:bg-green-800 disabled:cursor-not-allowed disabled:bg-gray-300"
                >
                    {loading ? "กำลังประมวลผล..." : "Start Segmentation"}
                </button>

                {/* แสดงผลการตรวจจับ */}
                {results && (
                    <div className="mt-6 rounded-xl border bg-white p-5">

                        <h2 className="mb-4 text-lg font-semibold text-gray-800">
                            Segmentation Results
                        </h2>

                        {/* สรุปจำนวนวัตถุแต่ละประเภท */}
                        <div className="grid grid-cols-3 gap-3">
                            {["Metal", "Paper", "Plastic"].map((name) => {
                                const count = results.filter(
                                    (item) => item.className === name
                                ).length;

                                return (
                                    <div
                                        key={name}
                                        className="rounded-lg bg-gray-50 p-4 text-center"
                                    >
                                        <p className="text-sm text-gray-500">
                                            {name}
                                        </p>

                                        <p className="mt-1 text-2xl font-bold text-green-700">
                                            {count}
                                        </p>
                                    </div>
                                );
                            })}
                        </div>

                        {/* รายละเอียดผลการตรวจจับ */}
                        <div className="mt-5">
                            <h3 className="mb-3 font-medium text-gray-800">
                                Detection Details
                            </h3>

                            {results.length === 0 ? (
                                <p className="text-sm text-gray-500">
                                    ไม่พบวัตถุที่ต้องการตรวจจับ
                                </p>
                            ) : (
                                results.map((item, index) => (
                                    <div
                                        key={index}
                                        className="flex justify-between border-b py-3 text-sm"
                                    >
                                        <span className="font-medium">
                                            {item.className}
                                        </span>

                                        <span className="text-gray-600">
                                            {(item.confidence * 100).toFixed(2)}%
                                        </span>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default Segmentation;
