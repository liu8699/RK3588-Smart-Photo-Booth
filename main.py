#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import cv2
import numpy as np
import mediapipe as mp

from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QSizePolicy, QGroupBox, QPushButton, QButtonGroup)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont


# 步进电机驱动（共阴极接法）

class GPIOStepperMotor:
    def __init__(self, pul_pin=104, dir_pin=115):
        self.pul_pin = pul_pin
        self.dir_pin = dir_pin
        
        self._export(self.pul_pin)
        self._export(self.dir_pin)
        
        self._set_direction(self.pul_pin, "out")
        self._set_direction(self.dir_pin, "out")
        
        self.pul_file = open(f"/sys/class/gpio/gpio{self.pul_pin}/value", "w")
        self.dir_file = open(f"/sys/class/gpio/gpio{self.dir_pin}/value", "w")
        print(" [GPIO Hardware] Motor Driver Ready. PUL=104, DIR=115")

    def _export(self, pin):
        if not os.path.exists(f"/sys/class/gpio/gpio{pin}"):
            try:
                with open("/sys/class/gpio/export", "w") as f:
                    f.write(str(pin))
            except Exception as e:
                print(f"GPIO Export Error ({pin}): {e}")

    def _set_direction(self, pin, direction):
        try:
            with open(f"/sys/class/gpio/gpio{pin}/direction", "w") as f:
                f.write(direction)
        except Exception as e:
            print(f"GPIO Direction Error ({pin}): {e}")

    def move(self, direction_action):
        if direction_action == "LOW":
            self.dir_file.write("0")
        elif direction_action == "HIGH":
            self.dir_file.write("1")
        self.dir_file.flush()

        steps = 60  # 随动步长微调
        delay = 0.0005
        
        for _ in range(steps):
            self.pul_file.write("1")
            self.pul_file.flush()
            time.sleep(delay)
            self.pul_file.write("0")
            self.pul_file.flush()
            time.sleep(delay)

    def close(self):
        self.pul_file.close()
        self.dir_file.close()



# 智能证件照一体机 PyQt5 主窗口

class SmartPhotoBooth(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能证件照自动对齐与合规系统 (AI抠图换底集成版)")
        self.setGeometry(100, 100, 1280, 800) 

        # 初始化参数配置默认值
        self.selected_type = "1寸"
        self.selected_bg = "蓝色"
        
        #  核心状态机控制变量
        self.is_preview_mode = True  # True: 实时拉流检测状态 | False: 锁定照片进行动态换底状态
        self.captured_frame = None   # 锁定抓拍的原图
        self.mask_3d = None          # 锁定计算的 AI 抠图掩膜
        
        # 标准证件照 BGR 颜色映射表
        self.colors = {
            "蓝色": (219, 142, 67),  # 标准证件蓝
            "红色": (46, 16, 200),   # 标准证件红
            "白色": (255, 255, 255)  # 纯白
        }
        
        # 自动拍摄倒计时核心状态机
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.handle_countdown)
        self.countdown_value = 3
        self.countdown_active = False
        self.photo_taken_lock = 0  # 恢复拍摄后的锁定帧数冷却器

        # 硬件与 AI 模型初始化
        self.motor = GPIOStepperMotor(pul_pin=104, dir_pin=115)
        self.cap = cv2.VideoCapture(21)
        if not self.cap.isOpened():
            print("❌ 错误：无法打开 21 号 USB 摄像头！")
            sys.exit(1)

        # 初始化看门狗人脸特征检测模型
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        #  初始化 MediaPipe 人像抠图分割模型
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=0)

        self.init_ui()

        # 主渲染定时器 (30 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

    def init_ui(self):
        #  
        self.setStyleSheet("""
            QMainWindow { background-color: #202225; }
            QLabel { color: #FFFFFF; }
            QGroupBox { 
                background-color: #2f3136; 
                border: 2px solid #4682B4; 
                border-radius: 8px; 
                margin-top: 15px;
                font-weight: bold;
                font-size: 14px;
                color: #4682B4;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton {
                background-color: #3b3e45;
                color: #DDDDDD;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                font-family: "微软雅黑";
            }
            QPushButton:checked {
                background-color: #4682B4;
                color: #FFFFFF;
                border: 1.5px solid #00FFFF;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: #1c1d1f;
                color: #666666;
                border: 1px solid #333;
            }
        """)

        # 主布局
        main_layout = QHBoxLayout()

        # ==================== 左侧：视频区域 + 底部看门狗状态栏 ====================
        left_layout = QVBoxLayout()
        
        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #151618; border: 2px solid #333; border-radius: 6px;")
        
        self.status_label = QLabel("正在拉起 AI 图像看门狗防御系统...", self)
        self.status_label.setFont(QFont("微软雅黑", 14, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #00FF00; background-color: #050505; padding: 15px; border-top: 3px solid #00FF00; border-radius: 4px;")
        
        left_layout.addWidget(self.video_label, stretch=6)
        left_layout.addWidget(self.status_label, stretch=1)
        main_layout.addLayout(left_layout, stretch=3)

        # ==================== 右侧：参数配置面板 + 动作控制 ====================
        right_layout = QVBoxLayout()

        # 1. 照片规格选择
        type_group = QGroupBox("照片规格选择 (Specification)", self)
        type_box = QVBoxLayout()
        self.btn_1inch = QPushButton("1寸证件照 (25x35mm)", self)
        self.btn_1inch.setCheckable(True)
        self.btn_1inch.setChecked(True)
        self.btn_2inch = QPushButton("2寸证件照 (35x53mm)", self)
        self.btn_2inch.setCheckable(True)
        self.btn_passport = QPushButton("护照签证照 (33x48mm)", self)
        self.btn_passport.setCheckable(True)

        self.type_group_btn = QButtonGroup(self)
        self.type_group_btn.addButton(self.btn_1inch)
        self.type_group_btn.addButton(self.btn_2inch)
        self.type_group_btn.addButton(self.btn_passport)
        self.type_group_btn.buttonClicked.connect(self.type_changed)

        type_box.addWidget(self.btn_1inch)
        type_box.addWidget(self.btn_2inch)
        type_box.addWidget(self.btn_passport)
        type_group.setLayout(type_box)
        right_layout.addWidget(type_group)

        # 2. 背景颜色选择
        bg_group = QGroupBox("背景颜色选择 (Background)", self)
        bg_box = QVBoxLayout()
        self.btn_blue = QPushButton("蓝色背景 (Blue)", self)
        self.btn_blue.setCheckable(True)
        self.btn_blue.setChecked(True)
        self.btn_red = QPushButton("红色背景 (Red)", self)
        self.btn_red.setCheckable(True)
        self.btn_white = QPushButton("白色背景 (White)", self)
        self.btn_white.setCheckable(True)

        self.bg_group_btn = QButtonGroup(self)
        self.bg_group_btn.addButton(self.btn_blue)
        self.bg_group_btn.addButton(self.btn_red)
        self.bg_group_btn.addButton(self.btn_white)
        self.bg_group_btn.buttonClicked.connect(self.bg_changed)

        bg_box.addWidget(self.btn_blue)
        bg_box.addWidget(self.btn_red)
        bg_box.addWidget(self.btn_white)
        bg_group.setLayout(bg_box)
        right_layout.addWidget(bg_group)

        # 3. 倒计时看板面板
        countdown_group = QGroupBox("全自动拍摄看门狗", self)
        countdown_box = QVBoxLayout()
        self.countdown_label = QLabel("等待合规", self)
        self.countdown_label.setFont(QFont("Impact", 38, QFont.Bold))
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet("color: #00FFFF; background-color: #101214; padding: 25px; border-radius: 6px; border: 1px solid #444;")
        countdown_box.addWidget(self.countdown_label)
        countdown_group.setLayout(countdown_box)
        right_layout.addWidget(countdown_group)

        # 🌟 4. 新增：后期抠图与保存动作控制面板
        action_group = QGroupBox("后期换底控制 (Action)", self)
        action_box = QVBoxLayout()
        
        self.btn_save = QPushButton(" 保存当前换底证件照", self)
        self.btn_save.setStyleSheet("background-color: #1565c0; color: white; font-weight: bold; font-size: 14px;")
        self.btn_save.clicked.connect(self.action_save)
        self.btn_save.setEnabled(False)  # 仅在拍照抠图成功后激活
        
        self.btn_retry = QPushButton(" 放弃并重新拍摄", self)
        self.btn_retry.clicked.connect(self.action_retry)
        self.btn_retry.setEnabled(False)  # 仅在拍照抠图成功后激活

        action_box.addWidget(self.btn_save)
        action_box.addWidget(self.btn_retry)
        action_group.setLayout(action_box)
        right_layout.addWidget(action_group)

        right_layout.addStretch()
        main_layout.addLayout(right_layout, stretch=1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def type_changed(self, button):
        self.selected_type = button.text().split(" ")[0]
        print(f"⚙️ 规格成功切换为: {self.selected_type}")

    def bg_changed(self, button):
        self.selected_bg = button.text()[:2]
        print(f"⚙️ 背景颜色成功切换为: {self.selected_bg}")

    def handle_countdown(self):
        if self.countdown_active:
            self.countdown_value -= 1
            if self.countdown_value > 0:
                self.countdown_label.setText(str(self.countdown_value))
            else:
                self.countdown_timer.stop()
                self.countdown_active = False
                self.countdown_label.setText("📸 💥")
                self.capture_photo()

    def capture_photo(self):
        """核心修改：拍照不再直接保存原图，而是触发 AI 抠图并锁定编辑"""
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("❌ 拍照失败：无法捕获摄像头画面！")
            return

        frame = cv2.flip(frame, 1)
        
        # 强刷提示，给用户提供视觉反馈
        self.status_label.setText("⚡ 触发合规抓拍！AI 正在进行高精度边缘人像剥离，请稍候...")
        self.status_label.setStyleSheet("color: #FFFF00; background-color: #050505; padding: 15px; border-top: 3px solid #FFFF00;")
        QApplication.processEvents()

        # 唤醒 MediaPipe Segmentation 进行推理
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(rgb_frame)
        
        if results.segmentation_mask is not None:
            raw_mask = results.segmentation_mask
            # 边缘高斯模糊羽化，平滑抠图硬边缘
            mask_blurred = cv2.GaussianBlur(raw_mask, (5, 5), 0)
            self.mask_3d = np.stack((mask_blurred,) * 3, axis=-1)
            
            #  状态机翻转：锁定抓拍帧，冻结摄像头和电机看门狗
            self.captured_frame = frame.copy()
            self.is_preview_mode = False
            
            # 激活保存与重拍动作按钮
            self.btn_save.setEnabled(True)
            self.btn_retry.setEnabled(True)
            
            # 更新提示面板
            self.countdown_label.setText("🎉")
            self.status_label.setText(f"🎉 智能抠图成功！请在右侧任意切换颜色查看效果。当前：{self.selected_bg}底")
            self.status_label.setStyleSheet("color: #00FFFF; background-color: #050505; padding: 15px; border-top: 3px solid #00FFFF;")
        else:
            self.status_label.setText("❌ 抠图失败：未能在画面中捕捉到清晰人像！")
            self.status_label.setStyleSheet("color: #FF3333; background-color: #050505; padding: 15px; border-top: 3px solid #FF3333;")
            self.action_retry()

    def action_retry(self):
        """放弃当前照片：清空数据，重启摄像头预览流与硬件检测"""
        self.is_preview_mode = True
        self.captured_frame = None
        self.mask_3d = None
        self.btn_save.setEnabled(False)
        self.btn_retry.setEnabled(False)
        
        # 🌟 恢复预览时赋予 75 帧(约2.5秒)冷却期，防止用户没调整好姿态直接再次秒触发拍摄
        self.photo_taken_lock = 75  
        self.countdown_active = False
        self.countdown_label.setText("等待合规")
        self.status_label.setText("安全熔断解除，正在重新拉起 AI 图像看门狗防御系统...")
        self.status_label.setStyleSheet("color: #00FF00; background-color: #050505; padding: 15px; border-top: 3px solid #00FF00;")

    def action_save(self):
        """合成干净的换底证件照并导出（不含任何 UI 提示信息）"""
        if self.captured_frame is not None and self.mask_3d is not None:
            h, w, _ = self.captured_frame.shape
            bg_image = np.zeros((h, w, 3), dtype=np.uint8)
            bg_image[:] = self.colors.get(self.selected_bg, (219, 142, 67))
            
            # Alpha 混合最终合成
            clean_output = (self.captured_frame * self.mask_3d + bg_image * (1.0 - self.mask_3d)).astype(np.uint8)
            
            save_dir = "/home/elf/Pictures/PhotoBooth"
            os.makedirs(save_dir, exist_ok=True)
            file_name = f"Matting_{self.selected_type}_{self.selected_bg}_{int(time.time())}.jpg"
            file_path = os.path.join(save_dir, file_name)
            
            cv2.imwrite(file_path, clean_output)
            print(f"💾 换底证件照成功保存至: {file_path}")
            
            self.status_label.setText(f"💾 保存成功！换底证件照已存入系统：{file_name}")
            self.status_label.setStyleSheet("color: #FFFF00; background-color: #050505; padding: 15px; border-top: 3px solid #FFFF00;")

    def show_opencv_in_qt(self, cv_img):
        """无损且不膨胀地将 OpenCV Mat 渲染到 QLabel 容器中"""
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        target_w = max(self.video_label.width(), 100)
        target_h = max(self.video_label.height(), 100)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(target_w, target_h, Qt.KeepAspectRatio))

    def update_frame(self):
        """核心调度器：完美融合实时检测流与静态抠图换底流"""
        #  拦截通道：如果处于非预览状态，代表正在进行换底编辑，立刻切断拉流与硬件逻辑
        if not self.is_preview_mode:
            if self.captured_frame is not None and self.mask_3d is not None:
                h, w, _ = self.captured_frame.shape
                bg_image = np.zeros((h, w, 3), dtype=np.uint8)
                bg_image[:] = self.colors.get(self.selected_bg, (219, 142, 67))
                
                # 30FPS 动态渲染当前选中的背景色合成效果
                output = (self.captured_frame * self.mask_3d + bg_image * (1.0 - self.mask_3d)).astype(np.uint8)
                self.show_opencv_in_qt(output)
            return

        # ---------------- 以下为原有 6.16 稳定版实时拉流与硬件看门狗判定 ----------------
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("⚠️ 错误：物理摄像头视频流突然中断！")
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        # 默认假设完全合规
        is_compliant = True
        status_text = "✅ 镜头前一切合规，准备拍摄！"
        status_color = "#00FF00"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 绘制特征脸部网格
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                # 获取关键点坐标
                nose_tip = face_landmarks.landmark[1]
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                left_eye_top = face_landmarks.landmark[159]
                left_eye_bottom = face_landmarks.landmark[145]
                right_eye_top = face_landmarks.landmark[386]
                right_eye_bottom = face_landmarks.landmark[374]

                # 歪头夹角判定
                tilt_angle = math.degrees(math.atan2(abs(right_eye.y - left_eye.y), abs(right_eye.x - left_eye.x)))
                
                if tilt_angle > 6.0:  
                    is_compliant = False
                    status_text = f"❌ 警告：检测到头部偏斜 ({tilt_angle:.1f}°)，请挺胸正头！"
                    status_color = "#FF3333"

                # 双眼闭眼检测
                elif abs(left_eye_top.y - left_eye_bottom.y) < 0.013 or abs(right_eye_top.y - right_eye_bottom.y) < 0.013:
                    is_compliant = False
                    status_text = "❌ 警告：请完全睁开双眼，避免眨眼瞬间！"
                    status_color = "#FF3333"

                # 步进电机物理滑台对齐高度随动
                elif nose_tip.y > 0.55:
                    is_compliant = False
                    status_text = " 状态：面部中线偏低 -> 物理滑台正在向下修正对齐..."
                    status_color = "#3399FF"
                    self.motor.move("LOW")
                elif nose_tip.y < 0.45:
                    is_compliant = False
                    status_text = " 状态：面部中线偏高 -> 物理滑台正在向上修正对齐..."
                    status_color = "#3399FF"
                    self.motor.move("HIGH")

        else:
            # 面部完全遮挡/丢失警报
            is_compliant = False
            status_text = "⚠️ 严重警报：面部特征不完整或被遮挡！请移开遮挡物并正对镜头！"
            status_color = "#FFFF00"

        # 全自动倒计时核心状态机控制
        if is_compliant:
            if self.photo_taken_lock > 0:
                status_text = f"⚙️ 防震荡锁保护中... 请稍候"
                status_color = "#00FF00"
            else:
                if not self.countdown_active:
                    self.countdown_active = True
                    self.countdown_value = 3
                    self.countdown_timer.start(1000)
                    self.countdown_label.setText(str(self.countdown_value))
                    self.countdown_label.setStyleSheet("color: #FF00FF; background-color: #101214; padding: 25px; border-radius: 6px; font-weight: bold;")
                status_text = f"📸 状态全合规！正在锁定姿态，将在 {self.countdown_value} 秒后自动拍照..."
        else:
            if self.photo_taken_lock == 0:
                if self.countdown_active:
                    self.countdown_timer.stop()
                    self.countdown_active = False
                self.countdown_label.setText("等待合规")
                self.countdown_label.setStyleSheet("color: #00FFFF; background-color: #101214; padding: 25px; border-radius: 6px;")

        if self.photo_taken_lock > 0:
            self.photo_taken_lock -= 1

        # 刷新看门狗状态看板
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(f"color: {status_color}; background-color: #050505; padding: 15px; border-top: 3px solid {status_color};")

        # 渲染当前帧
        self.show_opencv_in_qt(frame)

    def closeEvent(self, event):
        self.cap.release()
        self.motor.close()
        self.segmenter.close() # 释放人像分割模型
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SmartPhotoBooth()
    window.show()
    sys.exit(app.exec_())
