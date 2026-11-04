
import { useRef, useState } from "react";
import { loadModel, preprocessImage, processDetection } from "../utils/yolo";

function Detection() {
    // เก็บข้อมูลภาพ กล้อง และผลตรวจจับ
    const [image, setImage] = useState(null);
    const [camera, setCamera] = useState(false);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [imageSize, setImageSize] = useState(null);

    const videoRef = useRef(null);
    const streamRef = useRef(null);

    // ล้างภาพและผลตรวจจับเดิม
    const clearPreviousImage = () => {
        setImage(null);
        setResults(null);
        setImageSize(null);
    };

    // ประมวลผลภาพด้วยโมเดล YOLO
    const runDetection = async () => {
        if (!image) return;

        try {
            setLoading(true);
            setResults(null);

            // โหลดโมเดลและเตรียมภาพ
            const session = await loadModel("/models/detection_1.onnx");
            const img = new Image();
            img.src = image;

            await new Promise((resolve, reject) => {
                img.onload = resolve;
                img.onerror = reject;
            });

            const inputTensor = await preprocessImage(img);

            // ส่งภาพเข้าโมเดลและอ่านผลลัพธ์
            const output = await session.run({ images: inputTensor });

            console.log("Detection Output:", output.output0.dims);
            console.log("Detection Data:", output.output0.data);

            const detections = processDetection(output.output0);
            console.log("Detection Results:", detections);

            setResults(detections);
        } catch (error) {
            console.error("Detection Error:", error);
            alert("ไม่สามารถตรวจจับภาพได้");
        } finally {
            setLoading(false);
        }
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
        } catch {
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
        const canvas = document.createElement("canvas");

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext("2d").drawImage(video, 0, 0);

        setImage(canvas.toDataURL("image/jpeg"));
        closeCamera();
    };

    const getClassColor = (className) => {
        const colors = {
            Metal: "#ef4444",
            Paper: "#3b82f6",
            Plastic: "#eab308",
        };

        return colors[className] || "#22c55e";
    };

    return (
        <div className="space-y-5">
            <div>
                <h1 className="text-2xl font-bold">Object Detection</h1>
                <p className="mt-1 text-gray-500">
                    ตรวจจับขยะรีไซเคิลด้วย YOLO
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

            <div className="rounded-xl border bg-white p-6">
                <h2 className="mb-4 font-semibold">เลือกภาพ</h2>

                {/* อัปโหลดภาพหรือเปิดกล้อง */}
                <div className="flex flex-wrap gap-3">
                    <label className="cursor-pointer rounded-lg bg-green-700 px-4 py-2 text-white">
                        Upload Image
                        <input
                            type="file"
                            accept="image/*"
                            className="hidden"
                            onChange={(e) => {
                                const file = e.target.files[0];

                                if (file) {
                                    clearPreviousImage();
                                    setImage(URL.createObjectURL(file));
                                }

                                e.target.value = "";
                            }}
                        />
                    </label>

                    <button
                        onClick={openCamera}
                        className="rounded-lg border border-green-700 px-4 py-2 text-green-700"
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
                            className="w-full rounded-lg"
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
                                Close
                            </button>
                        </div>
                    </div>
                )}

                {/* แสดงภาพและกรอบตรวจจับ */}
                {image && (
                    <div className="mt-5">
                        <div className="relative inline-block max-w-full">
                            <img
                                src={image}
                                alt="Detection Preview"
                                onLoad={(e) => {
                                    setImageSize({
                                        width: e.target.naturalWidth,
                                        height: e.target.naturalHeight,
                                    });
                                }}
                                className="block max-h-96 max-w-full rounded-lg"
                            />

                            {results && imageSize && (
                                <svg
                                    viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
                                    className="absolute inset-0 h-full w-full"
                                >
                                    {results.map((item, index) => {
                                        const scaleX = imageSize.width / 640;
                                        const scaleY = imageSize.height / 640;

                                        const x = item.x1 * scaleX;
                                        const y = item.y1 * scaleY;
                                        const width = (item.x2 - item.x1) * scaleX;
                                        const height = (item.y2 - item.y1) * scaleY;

                                        return (
                                            <g key={index}>
                                                <rect
                                                    x={x}
                                                    y={y}
                                                    width={width}
                                                    height={height}
                                                    fill="none"
                                                    stroke={getClassColor(item.className)}
                                                    strokeWidth="3"
                                                />
                                                <text
                                                    x={x}
                                                    y={y - 5}
                                                    fill={getClassColor(item.className)}
                                                    fontSize="16"
                                                    fontWeight="bold"
                                                >
                                                    {item.className} {(item.confidence * 100).toFixed(1)}%
                                                </text>
                                            </g>
                                        );
                                    })}
                                </svg>
                            )}
                        </div>

                        {/* ลบภาพที่เลือก */}
                        <button
                            onClick={clearPreviousImage}
                            className="mt-3 rounded-lg bg-red-600 px-4 py-2 text-white hover:bg-red-700"
                        >
                            Remove Image
                        </button>
                    </div>
                )}

                {/* แสดงสรุปและรายละเอียดผลตรวจจับ */}
                {results && (
                    <div className="mt-6 rounded-xl border bg-white p-5">
                        <h2 className="mb-4 text-lg font-semibold">
                            Detection Results
                        </h2>

                        <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
                            {["Metal", "Paper", "Plastic"].map((name) => (
                                <div
                                    key={name}
                                    className="rounded-lg bg-gray-50 p-4"
                                >
                                    <p className="text-sm text-gray-500">{name}</p>
                                    <p className="mt-1 text-2xl font-bold">
                                        {results.filter((item) => item.className === name).length}
                                    </p>
                                    <p className="text-xs text-gray-400">
                                        Detected Objects
                                    </p>
                                </div>
                            ))}
                        </div>

                        <div className="overflow-x-auto">
                            <table className="w-full text-left text-sm">
                                <thead className="border-b bg-gray-50">
                                    <tr>
                                        <th className="p-3">No.</th>
                                        <th className="p-3">Object</th>
                                        <th className="p-3">Confidence</th>
                                    </tr>
                                </thead>

                                <tbody>
                                    {results.map((item, index) => (
                                        <tr key={index} className="border-b">
                                            <td className="p-3">{index + 1}</td>
                                            <td className="p-3 font-medium">
                                                {item.className}
                                            </td>
                                            <td className="p-3">
                                                {(item.confidence * 100).toFixed(2)}%
                                            </td>
                                        </tr>
                                    ))}

                                    {results.length === 0 && (
                                        <tr>
                                            <td colSpan="3" className="p-4 text-center text-gray-500">
                                                No objects detected
                                            </td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}

                {/* ปุ่มเริ่มตรวจจับ */}
                <button
                    onClick={runDetection}
                    disabled={!image || loading}
                    className="mt-5 rounded-lg bg-green-700 px-5 py-2 text-white disabled:bg-gray-300"
                >
                    {loading ? "กำลังตรวจจับ..." : "Start Detection"}
                </button>
            </div>
        </div>
    );
}

export default Detection;
