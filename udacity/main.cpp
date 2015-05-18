#include <stdlib.h>
#include <stdio.h>

void	max_min(float *h_values, size_t size, float &h_min, float &h_max); 

float			*init_values(size_t size)
{
  float			*values;

  if ((values = (float *)malloc(sizeof(float) * size)) != NULL) {
    for (size_t i = 0; i < size; i++)
      values[i] = float(i + 2);
    values[size / 2] = 100.f;
    values[size * 2 / 3] = 1.f;
    return (values);
  }
  return (NULL);
}


int			main(void)
{
  size_t		size = 90;
  int			bins = 8;
  float			*values = init_values(size);
  int			*hist;
  float			max = 0;
  float			min = 0;

  //  printf("Initial values :\n");
  //  for (size_t i = 0; i < size; i ++){
  //  printf("%.2f\n", values[i]);
  //  }

  max_min(values, size, min, max);
  
  printf("max = %f\n", max);
  printf("min = %f\n", min);

  return (0);
}
