package main

import (
	"flag"
	"fmt"
	"os"
	"path"

	"github.com/lazywei/go-opencv/opencv"
)

var (
	filename string
	result   string
	dir      string
)

func init() {
	flag.Parse()
	filename = flag.Arg(0)
	result = flag.Arg(0)

	dir = path.Dir(os.Args[0])
}

func main() {
	println(filename)
	println(dir)

	image := opencv.LoadImage(filename)
	if image == nil {
		panic("LoadImage fail")
	}
	defer image.Release()

	cascade := opencv.LoadHaarClassifierCascade(path.Join(dir, "haarcascade_frontalface_alt.xml"))
	faces := cascade.DetectObjects(image)

	for _, value := range faces {

		x := paddingXY(value.X())
		y := paddingXY(value.Y())
		w := paddingWH(value.Width())
		h := paddingWH(value.Height())

		x, y, w, h = square(image, x, y, w, h)

		fmt.Println(x, y, w, h)
		crop := opencv.Crop(image, x, y, w, h)
		crop = opencv.Resize(crop, 200, 200, 4)

		opencv.SaveImage("./tmp/"+filename, crop, 0)
		crop.Release()
		break
	}
}

func paddingXY(p int) int {
	p = p - int((float64(p) * 0.2))
	if p < 0 {
		p = 0
	}
	return p
}

func paddingWH(p int) int {
	p = p + int((float64(p) * 0.4))
	if p < 0 {
		p = 0
	}
	return p
}

func square(image *opencv.IplImage, x, y, w, h int) (int, int, int, int) {
	if w != h {
		if w > h {
			h = w
		} else {
			w = h
		}
	}

	if w+x > image.Width() {
		w = image.Width() - x
	}

	if h+y > image.Height() {
		h = image.Height() - y
	}

	if w != h {
		if w > h {
			w = h
		} else {
			h = w
		}
	}

	return x, y, w, h
}
