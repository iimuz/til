import { SummarizeA3rt } from '@/summarizeA3rt';

export class SummarizeRequest {
  public message: string = '';
}

export class SummarizeResponse {
  public summary: string = '';
}

export async function summarizeUseCase(request: SummarizeRequest): Promise<SummarizeResponse> {
  const summarizeAPI = new SummarizeA3rt();
  const summary = await summarizeAPI.summarize(request.message)
    .catch((err: any) => err);
  const response = new SummarizeResponse();
  response.summary = summary;
  return response;
}
