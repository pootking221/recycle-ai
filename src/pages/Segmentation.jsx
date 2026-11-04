
import { useRef, useState } from "react";
import * as ort from "onnxruntime-web";
import {
    loadModel,
    preprocessImage,
    createSegmentationOverlay,
} from "../utils/yolo";

function Segmentation() {
    const [image, setImage] = useState(null);
    const [camera, setCamera] = useState(false);

    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [maskImage, setMaskImage] = useState(null);

    const videoRef = useRef(null);
    const streamRef = useRef(null);

    const clearPreviousImage = () => {
        setImage(null);
        setMaskImage(null);
        setResults(null);
    };

    // ทดสอบการโหลดโมเดล
    // const testModel = async () => {
    //     try {
    //         console.log("กำลังโหลด Segmentation...");

    //         const session = await loadModel(
    //             "/models/segmentation.onnx"
    //         );

    //         console.log("Segmentation โหลดสำเร็จ!");
    //         console.log("Input Names:", session.inputNames);
    //         console.log("Output Names:", session.outputNames);

    //         const input = new ort.Tensor(
    //             "float32",
    //             new Float32Array(1 * 3 * 640 * 640),
    //             [1, 3, 640, 640]
    //         );

    //         const output = await session.run({
    //             images: input,
    //         });

    //         for (const [name, tensor] of Object.entries(output)) {
    //             console.log("Output:", name);
    //             console.log("Shape:", tensor.dims);
    //         }

    //         alert("ทดสอบ Segmentation สำเร็จ!");
    //     } catch (error) {
    //         console.error("Segmentation Error:", error);
    //         alert("ทดสอบ Segmentation ไม่สำเร็จ");
    //     }
    // };

    // เปิดกล้อง
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

    // ปิดกล้อง
    const closeCamera = () => {
        streamRef.current?.getTracks().forEach((track) => {
            track.stop();
        });

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

        const imageUrl = URL.createObjectURL(file);
        setImage(imageUrl);

        event.target.value = "";
    };

    // ประมวลผล Segmentation
    const runSegmentation = async () => {
        if (!image || loading) return;

        try {
            setLoading(true);
            setResults(null);
            setMaskImage(null);

            console.log("กำลังโหลด Segmentation Model...");

            const session = await loadModel(
                "/models/segmentation.onnx"
            );

            // โหลดภาพ
            const img = new Image();

            img.src = image;

            await new Promise((resolve, reject) => {
                img.onload = resolve;
                img.onerror = reject;
            });

            // เตรียมภาพสำหรับโมเดล
            const inputTensor = await preprocessImage(img);

            console.log("กำลังประมวลผลภาพ...");

            // รันโมเดล
            const output = await session.run({
                images: inputTensor,
            });

            console.log(
                "Segmentation Output 0:",
                output.output0.dims
            );

            console.log(
                "Segmentation Output 1:",
                output.output1.dims
            );

            // สร้าง Segmentation Mask
            const result = createSegmentationOverlay(
                output.output0,
                output.output1,
                img.naturalWidth,
                img.naturalHeight
            );

            console.log(
                "Segmentation Results:",
                result.detections
            );

            console.log(
                "จำนวนวัตถุที่ตรวจพบ:",
                result.count
            );

            // แสดงผล
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

            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold text-gray-800">
                    Polygon Segmentation
                </h1>

                <p className="mt-1 text-gray-500">
                    ตรวจจับและแสดงขอบเขตขยะรีไซเคิลด้วย YOLO
                </p>
            </div>

            {/* Main Card */}
            <div className="rounded-xl border bg-white p-6">

                <h2 className="mb-4 font-semibold text-gray-800">
                    เลือกภาพสำหรับประมวลผล
                </h2>

                {/* Buttons */}
                <div className="flex flex-wrap gap-3">

                    {/* Upload */}
                    <label className="cursor-pointer rounded-lg bg-green-700 px-4 py-2 text-white hover:bg-green-800">
                        Upload Image

                        <input
                            type="file"
                            accept="image/*"
                            className="hidden"
                            onChange={handleImageUpload}
                        />
                    </label>

                    {/* Camera */}
                    <button
                        onClick={openCamera}
                        className="rounded-lg border border-green-700 px-4 py-2 text-green-700 hover:bg-green-50"
                    >
                        Open Camera
                    </button>

                    {/* {/* Test Model */}
                    {/* <button
                        onClick={testModel}
                        className="rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700"
                    >
                        Test Segmentation Model
                    </button>  */}

                </div>

                {/* Camera Preview */}
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

                {/* Image Preview */}
                {image && (
                    <div className="mt-6">

                        <h2 className="mb-3 font-semibold text-gray-800">
                            Segmentation Preview
                        </h2>

                        <div className="relative inline-block max-w-full overflow-hidden rounded-lg bg-gray-100">

                            {/* Original Image */}
                            <img
                                src={image}
                                alt="Original Image"
                                className="block max-h-96 max-w-full"
                            />

                            {/* Segmentation Mask */}
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

                        <button
                            onClick={() => {
                                clearPreviousImage();
                            }}
                            className="mt-3 rounded-lg bg-red-600 px-4 py-2 text-white hover:bg-red-700"
                        >
                            Remove Image
                        </button>
                    </div>

                )}

                {/* Start Segmentation */}
                <button
                    onClick={runSegmentation}
                    disabled={!image || loading}
                    className="mt-5 rounded-lg bg-green-700 px-5 py-2 text-white hover:bg-green-800 disabled:cursor-not-allowed disabled:bg-gray-300"
                >
                    {loading
                        ? "กำลังประมวลผล..."
                        : "Start Segmentation"}
                </button>

                {/* Results */}
                {results && (
                    <div className="mt-6 rounded-xl border bg-white p-5">

                        <h2 className="mb-4 text-lg font-semibold text-gray-800">
                            Segmentation Results
                        </h2>

                        {/* Class Counts */}
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

                        {/* Detection Details */}
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
                                            {(item.score * 100).toFixed(2)}%
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
