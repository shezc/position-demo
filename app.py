"""
图片文本检测和高亮显示工具 - Web版本
基于Flask的Web应用
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import easyocr
from pathlib import Path
import os
import base64
import io
from werkzeug.utils import secure_filename
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading

app = Flask(__name__)
CORS(app)

# 配置
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB最大文件大小
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 创建必要的文件夹
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# 全局OCR读取器（延迟初始化）
reader = None


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_ocr(languages=['ch_sim', 'en']):
    """初始化OCR引擎"""
    global reader
    if reader is None:
        print("正在初始化OCR引擎...")
        reader = easyocr.Reader(languages, gpu=False)
        print("OCR引擎初始化完成！")
    return reader


def detect_text(image):
    """检测图片中的所有文本"""
    global reader
    if reader is None:
        reader = init_ocr()
    
    results = reader.readtext(image)
    
    text_list = []
    for (bbox, text, confidence) in results:
        # 将bbox从numpy数组转换为Python列表，确保所有数值都是Python原生类型
        bbox_list = [[float(point[0]), float(point[1])] for point in bbox]
        
        text_list.append({
            'bbox': bbox_list,
            'text': text,
            'confidence': float(confidence)
        })
    
    return text_list


def calculate_keyword_bbox(bbox, text, target_text):
    """
    计算关键字在文本中对应的bbox区域
    根据关键字在文本中的位置和长度，按比例缩小bbox
    """
    text_lower = text.lower()
    target_lower = target_text.lower()
    
    # 如果完全匹配，返回整个bbox
    if text_lower == target_lower:
        return bbox
    
    # 查找关键字在文本中的位置
    start_idx = text_lower.find(target_lower)
    if start_idx == -1:
        # 如果找不到，返回整个bbox
        return bbox
    
    end_idx = start_idx + len(target_text)
    text_length = len(text)
    
    # bbox是4个点的列表，通常格式为 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    bbox_array = np.array(bbox, dtype=np.float32)
    
    x_coords = bbox_array[:, 0]
    y_coords = bbox_array[:, 1]
    
    # 更准确地找到四个角点
    # 左上：x最小，y最小
    top_left_idx = np.argmin(x_coords + y_coords)
    top_left = bbox_array[top_left_idx]
    
    # 右上：x最大，y最小
    top_right_idx = np.argmax(x_coords - y_coords)
    top_right = bbox_array[top_right_idx]
    
    # 右下：x最大，y最大
    bottom_right_idx = np.argmax(x_coords + y_coords)
    bottom_right = bbox_array[bottom_right_idx]
    
    # 左下：x最小，y最大
    bottom_left_idx = np.argmin(x_coords - y_coords)
    bottom_left = bbox_array[bottom_left_idx]
    
    # 计算文本行的总宽度
    top_width = np.linalg.norm(top_right - top_left)
    bottom_width = np.linalg.norm(bottom_right - bottom_left)
    
    # 计算每个字符的实际显示宽度（考虑中文字符和英文字符宽度不同）
    # 中文字符通常宽度约为英文字符的2倍
    def get_char_display_width(char):
        # 判断是否为中文字符（包括中文标点）
        if '\u4e00' <= char <= '\u9fff' or '\u3000' <= char <= '\u303f':
            return 2.0  # 中文字符宽度为2
        else:
            return 1.0  # 英文字符宽度为1
    
    # 计算文本的总显示宽度
    total_display_width = sum(get_char_display_width(char) for char in text)
    
    # 计算关键字之前的显示宽度
    prefix_display_width = sum(get_char_display_width(text[i]) for i in range(start_idx))
    
    # 计算关键字的显示宽度
    keyword_display_width = sum(get_char_display_width(text[i]) for i in range(start_idx, end_idx))
    
    # 计算关键字在文本中的显示位置比例
    start_ratio = prefix_display_width / total_display_width if total_display_width > 0 else 0
    end_ratio = (prefix_display_width + keyword_display_width) / total_display_width if total_display_width > 0 else 1
    
    # 计算顶部边界上的关键字位置
    top_vector = top_right - top_left
    top_keyword_start = top_left + top_vector * start_ratio
    top_keyword_end = top_left + top_vector * end_ratio
    
    # 计算底部边界上的关键字位置
    bottom_vector = bottom_right - bottom_left
    bottom_keyword_start = bottom_left + bottom_vector * start_ratio
    bottom_keyword_end = bottom_left + bottom_vector * end_ratio
    
    # 构建关键字区域的bbox
    keyword_bbox = np.array([
        top_keyword_start,      # 左上
        top_keyword_end,         # 右上
        bottom_keyword_end,      # 右下
        bottom_keyword_start     # 左下
    ], dtype=np.float32)
    
    return keyword_bbox


def highlight_single_keyword(image, target_text, text_list, highlight_color=(0, 255, 255), alpha=0.4):
    """
    为单个关键字生成高亮图片
    """
    highlighted_image = image.copy()
    overlay = image.copy()
    color = highlight_color
    keyword_matches = []
    
    # 查找匹配的文本
    for item in text_list:
        text = item['text']
        # 支持部分匹配和完全匹配
        if target_text.lower() in text.lower() or text.lower() in target_text.lower():
            bbox = item['bbox']
            
            # 计算关键字对应的bbox区域
            keyword_bbox = calculate_keyword_bbox(bbox, text, target_text)
            
            # 将bbox转换为整数坐标
            keyword_bbox_int = np.array(keyword_bbox, dtype=np.int32)
            
            # 只绘制半透明填充背景，不绘制边框和文本标签
            cv2.fillPoly(overlay, [keyword_bbox_int], color)
            cv2.addWeighted(overlay, alpha, highlighted_image, 1 - alpha, 0, highlighted_image)
            
            # 记录匹配结果
            keyword_matches.append({
                'text': item['text'],
                'confidence': item['confidence'],
                'keyword': target_text
            })
    
    return highlighted_image, keyword_matches


def highlight_text(image, target_texts, text_list, highlight_color=(0, 255, 255), 
                   alpha=0.4, selected_keyword=None):
    """
    在图片上高亮显示目标文本（仅高亮关键字部分）
    支持多个关键字，使用统一颜色
    如果指定selected_keyword，只高亮该关键字
    如果没有指定selected_keyword，并发处理所有关键字
    """
    # 如果target_texts是字符串，转换为列表
    if isinstance(target_texts, str):
        # 支持中文逗号、英文逗号、分号、换行符分隔
        target_texts = [t.strip() for t in target_texts.replace('，', ',').replace('\n', ',').replace(';', ',').split(',') if t.strip()]
    
    matches_by_keyword = {}  # 按关键字分组存储匹配结果
    images_by_keyword = {}   # 按关键字存储高亮图片
    
    # 如果指定了selected_keyword，只处理该关键字
    if selected_keyword:
        if selected_keyword in target_texts:
            highlighted_image, keyword_matches = highlight_single_keyword(
                image, selected_keyword, text_list, highlight_color, alpha
            )
            matches_by_keyword[selected_keyword] = keyword_matches
            images_by_keyword[selected_keyword] = highlighted_image
    else:
        # 并发处理所有关键字
        with ThreadPoolExecutor(max_workers=min(len(target_texts), 8)) as executor:
            futures = {}
            for target_text in target_texts:
                if not target_text:
                    continue
                future = executor.submit(
                    highlight_single_keyword, 
                    image.copy(),  # 每个线程使用图片副本
                    target_text, 
                    text_list, 
                    highlight_color, 
                    alpha
                )
                futures[target_text] = future
            
            # 收集结果
            for target_text, future in futures.items():
                highlighted_image, keyword_matches = future.result()
                matches_by_keyword[target_text] = keyword_matches
                images_by_keyword[target_text] = highlighted_image
    
    # 合并所有匹配结果
    all_matches = []
    for keyword, matches in matches_by_keyword.items():
        all_matches.extend(matches)
    
    # 返回第一个关键字的高亮图片（用于默认显示）
    default_image = images_by_keyword.get(target_texts[0] if target_texts else None, image)
    
    return default_image, all_matches, matches_by_keyword, images_by_keyword


def image_to_base64(image):
    """将OpenCV图片转换为base64字符串"""
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{image_base64}"


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/highlight', methods=['POST'])
def highlight():
    """高亮显示目标文本API"""
    try:
        # 检查是否有文件
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件格式'}), 400
        
        # 获取参数
        target_text = request.form.get('target_text', '').strip()
        if not target_text:
            return jsonify({'error': '请输入目标文本'}), 400
        
        # 解析多个关键字（支持中文逗号、英文逗号、分号、换行符分隔）
        target_texts = [t.strip() for t in target_text.replace('，', ',').replace('\n', ',').replace(';', ',').split(',') if t.strip()]
        if not target_texts:
            return jsonify({'error': '请输入有效的目标文本'}), 400
        
        # 获取选中的关键字（用于切换显示）
        selected_keyword = request.form.get('selected_keyword', '').strip()
        if selected_keyword == '':
            selected_keyword = None
        
        languages = request.form.get('languages', 'ch_sim,en').split(',')
        
        # 保存上传的文件
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # 读取图片
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': '无法读取图片文件'}), 400
        
        # 初始化OCR（如果需要）
        init_ocr(languages)
        
        # 检测文本
        text_list = detect_text(image)
        
        # 高亮显示（支持多个关键字，统一颜色，并发处理）
        highlighted_image, matches, matches_by_keyword, images_by_keyword = highlight_text(
            image, target_texts, text_list, selected_keyword=selected_keyword
        )
        
        # 转换为base64（默认显示第一个关键字的结果）
        result_base64 = image_to_base64(highlighted_image)
        
        # 将所有关键字的高亮图片转换为base64并缓存
        keyword_images = {}
        for keyword, keyword_image in images_by_keyword.items():
            keyword_images[keyword] = image_to_base64(keyword_image)
        
        # 清理临时文件
        os.remove(filepath)
        
        # 按关键字组织匹配结果
        keyword_results = {}
        for keyword, keyword_matches in matches_by_keyword.items():
            keyword_results[keyword] = {
                'matches': [
                    {
                        'text': m['text'],
                        'confidence': m['confidence']
                    }
                    for m in keyword_matches
                ],
                'count': len(keyword_matches)
            }
        
        return jsonify({
            'success': True,
            'image': result_base64,  # 默认显示的图片
            'keyword_images': keyword_images,  # 所有关键字的高亮图片（缓存）
            'matches': [
                {
                    'text': m['text'],
                    'confidence': m['confidence'],
                    'keyword': m.get('keyword', '')
                }
                for m in matches
            ],
            'match_count': len(matches),
            'total_text_count': len(text_list),
            'keywords': target_texts,
            'keyword_results': keyword_results,
            'selected_keyword': selected_keyword
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查API"""
    return jsonify({
        'status': 'ok',
        'ocr_ready': reader is not None
    })


if __name__ == '__main__':
    print("=" * 50)
    print("图片文本检测和高亮工具 - Web版本")
    print("=" * 50)
    print("\n服务器启动中...")
    print("访问 http://localhost:5000 使用Web界面")
    print("\n注意：首次使用时会自动初始化OCR引擎，可能需要一些时间")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
