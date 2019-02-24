import { summarizeUseCase, SummarizeRequest, SummarizeResponse } from '@/summarizeUseCase';

export async function summarizeController(message: string): Promise<string> {
  const request = new SummarizeRequest();
  request.message = message;
  const response = await summarizeUseCase(request).catch((err: any) => err);
  return response.summary;
}

