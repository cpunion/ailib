package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	ailibmodel "github.com/cpunion/ailib/adk/model"
	"google.golang.org/genai"
)

func main() {
	var (
		modelSpec   string
		textPrompt  string
		imagePrompt string
		outDir      string
		reference   string
		aspect      string
	)

	flag.StringVar(&modelSpec, "model", "gemini:gemini-3.1-flash-image-preview", "provider:model")
	flag.StringVar(&textPrompt, "text", "", "text prompt")
	flag.StringVar(&imagePrompt, "image", "", "image prompt")
	flag.StringVar(&outDir, "out", "", "output directory")
	flag.StringVar(&reference, "reference", "", "optional reference image path")
	flag.StringVar(&aspect, "aspect", "16:9", "image aspect ratio")
	flag.Parse()

	if strings.TrimSpace(textPrompt) == "" && strings.TrimSpace(imagePrompt) == "" {
		exitf("至少提供 --text 或 --image")
	}

	if strings.TrimSpace(outDir) == "" {
		tmpDir, err := os.MkdirTemp("", "ailib-multimodal-demo-*")
		if err != nil {
			exitf("创建临时目录失败: %v", err)
		}
		outDir = tmpDir
	}
	if err := os.MkdirAll(outDir, 0755); err != nil {
		exitf("创建输出目录失败: %v", err)
	}

	ctx := context.Background()
	if strings.TrimSpace(textPrompt) != "" {
		result, err := ailibmodel.GenerateOnce(ctx, modelSpec, ailibmodel.NewTextRequest(textPrompt, nil), nil)
		if err != nil {
			exitf("文本生成失败: %v", err)
		}
		textPath := filepath.Join(outDir, "text.txt")
		if err := os.WriteFile(textPath, []byte(strings.TrimSpace(result.Text)+"\n"), 0644); err != nil {
			exitf("写入文本结果失败: %v", err)
		}
		fmt.Printf("text: %s\n", textPath)
	}

	if strings.TrimSpace(imagePrompt) != "" {
		references, err := loadReferences(reference)
		if err != nil {
			exitf("读取参考图失败: %v", err)
		}
		config := &genai.GenerateContentConfig{
			ResponseModalities: []string{"IMAGE"},
			ImageConfig:        &genai.ImageConfig{AspectRatio: strings.TrimSpace(aspect)},
		}
		result, err := ailibmodel.GenerateOnce(ctx, modelSpec, ailibmodel.NewImageRequest(imagePrompt, config, references...), nil)
		if err != nil {
			exitf("图片生成失败: %v", err)
		}
		if len(result.Images) == 0 {
			exitf("图片生成未返回任何图片")
		}
		for i, image := range result.Images {
			path := filepath.Join(outDir, fmt.Sprintf("image_%02d%s", i+1, fileExtFromMIME(image.MIMEType)))
			if err := os.WriteFile(path, image.Data, 0644); err != nil {
				exitf("写入图片结果失败: %v", err)
			}
			fmt.Printf("image: %s\n", path)
		}
	}

	fmt.Printf("out: %s\n", outDir)
}

func loadReferences(path string) ([]*genai.Blob, error) {
	refPath := strings.TrimSpace(path)
	if refPath == "" {
		return nil, nil
	}
	data, err := os.ReadFile(refPath)
	if err != nil {
		return nil, err
	}
	return []*genai.Blob{{
		MIMEType: detectImageMimeType(refPath),
		Data:     data,
	}}, nil
}

func detectImageMimeType(path string) string {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".webp":
		return "image/webp"
	default:
		return "image/png"
	}
}

func fileExtFromMIME(mimeType string) string {
	switch strings.ToLower(strings.TrimSpace(mimeType)) {
	case "image/jpeg":
		return ".jpg"
	case "image/webp":
		return ".webp"
	default:
		return ".png"
	}
}

func exitf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
