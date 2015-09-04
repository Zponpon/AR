#include "Frame.h"

bool KeyFrameSelection(unsigned long index, std::vector<KeyFrame> &keyFrames)
{
	if (index - keyFrames[0].index < 20)
		return false;
	return true;
}